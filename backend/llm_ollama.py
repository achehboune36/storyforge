"""Ollama streaming client.

Notes that cost time to learn, recorded so they are not re-learned:

  * Ollama runs as the **Windows** application here, not inside WSL. Mirrored
    WSL networking makes it reachable on 127.0.0.1:11434 anyway.
  * `num_ctx` MUST be sent explicitly. Ollama's default is 4096 and it
    **truncates silently** — a long story would quietly lose its beginning with
    no error and no log line.
  * `keep_alive: 0` evicts immediately (measured 0.30s); `-1` pins forever.
    That asymmetry is the whole basis of the GPU strategy: cycling the LLM
    costs ~4s a turn, cycling ComfyUI would cost ~25s.
  * Responses are newline-delimited JSON objects, not SSE. There is no
    `data: ` prefix and no `[DONE]` sentinel — the final object has `done:true`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Iterable

import aiohttp

from backend.turn_parser import Event, ProseSplitter, Stats

ChunkCb = Callable[[str], Awaitable[None]]


@dataclass
class LlmResult:
    ok: bool
    text: str = ""
    chunks: int = 0
    t_first_token_s: float | None = None
    total_s: float = 0.0
    eval_count: int | None = None        # tokens generated, per Ollama
    prompt_eval_count: int | None = None  # tokens of prompt consumed
    truncated: bool = False
    error: str | None = None

    @property
    def chunks_per_s(self) -> float:
        decode = self.total_s - (self.t_first_token_s or 0.0)
        return self.chunks / decode if decode > 0 else 0.0


class OllamaClient:
    def __init__(self, cfg: dict):
        c = cfg["llm"]
        self.host = c["host"].rstrip("/")
        self.model = c["model"]
        self.num_ctx = int(c["num_ctx"])
        self.num_predict = int(c.get("num_predict", 700))
        self.keep_alive_active = c.get("keep_alive_active", "10m")
        self._cfg = c

    # ------------------------------------------------------------------ health

    async def is_up(self) -> bool:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.host}/api/tags",
                                 timeout=aiohttp.ClientTimeout(total=5)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def available_models(self) -> list[str]:
        try:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.host}/api/tags",
                                 timeout=aiohttp.ClientTimeout(total=10)) as r:
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
        keep_alive: str | int | None = None,
        num_predict: int | None = None,
        temperature: float | None = None,
        timeout_s: float = 600.0,
    ) -> LlmResult:
        body = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": keep_alive if keep_alive is not None else self.keep_alive_active,
            "options": {
                "num_ctx": self.num_ctx,
                "num_predict": num_predict if num_predict is not None else self.num_predict,
            },
        }
        if temperature is not None:
            body["options"]["temperature"] = temperature

        res = LlmResult(ok=False)
        parts: list[str] = []
        t0 = time.time()
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.host}/api/chat", json=body,
                                  timeout=aiohttp.ClientTimeout(total=timeout_s)) as r:
                    if r.status != 200:
                        res.error = f"ollama {r.status}: {(await r.text())[:300]}"
                        return res
                    async for raw in r.content:
                        if not raw.strip():
                            continue
                        try:
                            m = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        if "error" in m:
                            res.error = str(m["error"])[:300]
                            return res
                        piece = (m.get("message") or {}).get("content", "")
                        if piece:
                            if res.t_first_token_s is None:
                                res.t_first_token_s = time.time() - t0
                            res.chunks += 1
                            parts.append(piece)
                            if on_chunk:
                                await on_chunk(piece)
                        if m.get("done"):
                            res.eval_count = m.get("eval_count")
                            res.prompt_eval_count = m.get("prompt_eval_count")
                            # "length" means num_predict cut it off mid-sentence.
                            res.truncated = m.get("done_reason") == "length"
                            break
            res.ok = True
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"

        res.text = "".join(parts)
        res.total_s = time.time() - t0
        return res

    # -------------------------------------------------------------- vram control

    async def evict(self) -> bool:
        """Release VRAM now. Measured at ~0.30s."""
        try:
            async with aiohttp.ClientSession() as s:
                async with s.post(f"{self.host}/api/chat", json={
                    "model": self.model, "messages": [],
                    "keep_alive": 0, "stream": False,
                }, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    return r.status == 200
        except Exception:
            return False

    async def warm(self) -> float:
        """Load the model without generating. Returns seconds taken."""
        t0 = time.time()
        await self.stream([{"role": "user", "content": "hi"}], num_predict=1)
        return time.time() - t0


# --------------------------------------------------------------------- turns

@dataclass
class TurnResult:
    llm: LlmResult
    events: list[Event] = field(default_factory=list)
    stats: Stats = field(default_factory=Stats)

    @property
    def ok(self) -> bool:
        return self.llm.ok


async def stream_turn(
    client: OllamaClient,
    messages: list[dict],
    *,
    on_event: Callable[[Event], Awaitable[None]] | None = None,
    max_images: int = 2,
    previous_keywords: str | None = None,
    **stream_kw,
) -> TurnResult:
    """Stream one narrative turn, splitting prose from image directives live.

    `on_event` fires as each ProseChunk / ImageSlot becomes available, which is
    what lets the UI open an image placeholder at the right position *while*
    the prose is still arriving.
    """
    splitter = ProseSplitter(max_images=max_images,
                             previous_keywords=previous_keywords)
    collected: list[Event] = []

    async def on_chunk(piece: str) -> None:
        for ev in splitter.feed(piece):
            collected.append(ev)
            if on_event:
                await on_event(ev)

    res = await client.stream(messages, on_chunk=on_chunk, **stream_kw)

    for ev in splitter.finish():
        collected.append(ev)
        if on_event:
            await on_event(ev)

    return TurnResult(llm=res, events=collected, stats=splitter.stats)


def build_messages(system: str, history: Iterable[dict], user: str) -> list[dict]:
    """Assemble the /api/chat message list. History is already budgeted."""
    msgs = [{"role": "system", "content": system}]
    msgs.extend(history)
    msgs.append({"role": "user", "content": user})
    return msgs


# ------------------------------------------------------------- repair path

REPAIR_INSTRUCTION = (
    "Below is a passage from an illustrated story. If it moves to a place or "
    "person that would be worth showing as a picture, reply with ONLY a "
    "comma-separated list of 6-12 visual keywords naming what is visible "
    "there - place, objects, light, colour. No sentences, no actions, no "
    "explanation. If the passage is only conversation or a small action in a "
    "place already described, reply with the single word NONE."
)


async def request_keywords(client: OllamaClient, passage: str, *,
                           timeout_s: float = 60.0) -> str | None:
    """Ask for image keywords when the inline directive did not appear.

    Directive compliance from llama3.2:3b is probabilistic - roughly 50-75% at
    default temperature, with the same prompt producing a directive on one run
    and not the next. Prompt shape was tuned as far as it usefully goes (see
    stories/noir.json), so the remaining gap is closed here.

    Cheap because the model is already resident: one short call, low
    temperature, ~60 tokens. Costs ~1s against a ~14s image.

    Returns keywords, or None when the model judges no illustration is due -
    which is a real answer, not a failure.
    """
    res = await client.stream(
        [{"role": "user", "content": f"{REPAIR_INSTRUCTION}\n\nPASSAGE:\n{passage}"}],
        num_predict=60, temperature=0.3, timeout_s=timeout_s)
    if not res.ok:
        return None
    text = res.text.strip().strip('"').strip()
    if not text or text.upper().startswith("NONE"):
        return None
    # A compliant reply is one line of comma-separated keywords. Anything with
    # sentence punctuation is the model explaining itself; take the first line
    # and drop it if it still does not look like a keyword list.
    first = text.splitlines()[0].strip()
    if first.count(",") < 2 or len(first) > 300:
        return None
    return first
