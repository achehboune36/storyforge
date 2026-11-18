"""Ollama Cloud narrator, OpenAI-compatible endpoint.

Same surface as OllamaClient so session.py does not care which is in use.

Measured on deepseek-v4-pro (peak rates: input $1.32/M, cached input $0.044/M,
output $3.96/M):

    cold   prompt 555 (0 cached)    completion 361   $0.00216
    warm   prompt 555 (554 cached)  completion 367   $0.00148

Two facts shape this module:

  * **Cached input is 30x cheaper than fresh input.** That only pays if the
    system prompt is byte-identical turn to turn. Anything varying inside it -
    a clock, a re-ordered roster - misses the cache and every turn pays full
    price for the whole prompt. So the per-turn clock and scene state go in the
    *user* message, never the system block.
  * **These are reasoning models.** They emit a `reasoning` field before any
    `content`, billed as output at 3x the input rate. A max_tokens that does
    not clear the thinking returns empty content with finish_reason="length" -
    which reads exactly like a refusal and is not one. Caps must be generous
    and truncation must be reported, not scored.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Iterable

import aiohttp

from backend.llm_ollama import LlmResult

ChunkCb = Callable[[str], Awaitable[None]]

# Peak rates, USD per token. Off-peak is half.
RATE_IN, RATE_CACHED, RATE_OUT = 1.32e-6, 0.044e-6, 3.96e-6


def load_key() -> str:
    """Key from env, else .env beside the project. Never from a tracked file."""
    key = os.environ.get("OLLAMA_API_KEY", "").strip()
    if key:
        return key
    envf = Path(__file__).resolve().parent.parent / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("OLLAMA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


@dataclass
class Spend:
    turns: int = 0
    usd: float = 0.0
    last_usd: float = 0.0
    last_prompt: int = 0
    last_cached: int = 0
    last_output: int = 0

    def add(self, prompt: int, cached: int, output: int) -> float:
        fresh = max(prompt - cached, 0)
        cost = fresh * RATE_IN + cached * RATE_CACHED + output * RATE_OUT
        self.turns += 1
        self.usd += cost
        self.last_usd, self.last_prompt = cost, prompt
        self.last_cached, self.last_output = cached, output
        return cost


class CloudClient:
    """Drop-in replacement for OllamaClient against Ollama Cloud."""

    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.host = c.get("cloud_host", "https://ollama.com").rstrip("/")
        self.model = c["model"]
        self.num_ctx = int(c.get("num_ctx", 16000))
        self.num_predict = int(c.get("num_predict", 3000))
        self.key = load_key()
        self.budget_usd = float(c.get("budget_per_turn_usd", 0.03))
        self.spend = Spend()

    # ------------------------------------------------------------------ health

    async def is_up(self) -> bool:
        if not self.key:
            return False
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.host}/api/tags",
                                 headers={"Authorization": f"Bearer {self.key}"},
                                 timeout=aiohttp.ClientTimeout(total=10)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def available_models(self) -> list[str]:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.host}/api/tags",
                                 headers={"Authorization": f"Bearer {self.key}"},
                                 timeout=aiohttp.ClientTimeout(total=15)) as r:
                    d = await r.json()
            return [m["name"] for m in d.get("models", [])]
        except Exception:
            return []

    # ------------------------------------------------------------------ stream

    async def stream(
        self,
        messages: list[dict],
        *,
        on_chunk: ChunkCb | None = None,
        keep_alive: str | int | None = None,   # accepted and ignored
        num_predict: int | None = None,
        temperature: float | None = None,
        timeout_s: float = 300.0,
    ) -> LlmResult:
        res = LlmResult(ok=False)
        if not self.key:
            res.error = "no OLLAMA_API_KEY (env or .env)"
            return res

        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_tokens": num_predict or self.num_predict,
            "temperature": 0.95 if temperature is None else temperature,
            "stream_options": {"include_usage": True},
        }
        parts: list[str] = []
        t0 = time.time()
        usage: dict = {}
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(
                    f"{self.host}/v1/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.key}",
                             "Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=timeout_s),
                ) as r:
                    if r.status != 200:
                        res.error = f"cloud {r.status}: {(await r.text())[:300]}"
                        return res
                    async for raw in r.content:
                        line = raw.decode(errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            m = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        if m.get("usage"):
                            usage = m["usage"]
                        for ch in m.get("choices", []):
                            delta = ch.get("delta") or {}
                            # `reasoning` arrives first and is not story text.
                            piece = delta.get("content") or ""
                            if piece:
                                if res.t_first_token_s is None:
                                    res.t_first_token_s = time.time() - t0
                                res.chunks += 1
                                parts.append(piece)
                                if on_chunk:
                                    await on_chunk(piece)
                            if ch.get("finish_reason") == "length":
                                # With no content at all, reasoning ate the
                                # whole budget. With content, the prose was
                                # cut mid-sentence — which used to pass
                                # silently and simply ended the story in the
                                # middle of a word.
                                res.truncated = True
                                if not parts:
                                    res.error = ("budget exhausted during reasoning — "
                                                 "raise num_predict")
            res.ok = bool(parts) or not res.error
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"

        res.text = "".join(parts)
        res.total_s = time.time() - t0
        res.prompt_eval_count = usage.get("prompt_tokens")
        res.eval_count = usage.get("completion_tokens")

        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        cost = self.spend.add(usage.get("prompt_tokens", 0), cached,
                              usage.get("completion_tokens", 0))
        if cost > self.budget_usd:
            print(f"[cloud] turn cost ${cost:.4f} EXCEEDS budget "
                  f"${self.budget_usd:.3f} (prompt {usage.get('prompt_tokens')}, "
                  f"cached {cached}, out {usage.get('completion_tokens')})", flush=True)
        return res

    # ----------------------------------------------------- vram no-ops

    async def evict(self) -> bool:
        """Nothing resident locally — the GPU is entirely the image model's."""
        return True

    async def warm(self) -> float:
        return 0.0

    def cost_snapshot(self) -> dict:
        s = self.spend
        return {
            "model": self.model, "turns": s.turns,
            "session_usd": round(s.usd, 5),
            "last_turn_usd": round(s.last_usd, 5),
            "budget_per_turn_usd": self.budget_usd,
            "last_prompt_tokens": s.last_prompt,
            "last_cached_tokens": s.last_cached,
            "last_output_tokens": s.last_output,
        }


def make_llm(cfg: dict):
    """Pick cloud or local from config."""
    if cfg["llm"].get("provider") == "cloud":
        return CloudClient(cfg)
    from backend.llm_ollama import OllamaClient
    return OllamaClient(cfg)
