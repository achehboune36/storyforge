"""Benchmark client for ComfyUI — prototype of backend/comfy_client.py.

Implements the protocol details verified against ComfyUI v0.11.1:

  * WebSocket is opened BEFORE POSTing /prompt, so a cached or instant job
    cannot finish before we are listening. (Ported from game-poc.)
  * The feature_flags handshake MUST be the first frame on the socket, or no
    preview frames are sent at all — comfy_execution/progress.py:207-231 gates
    them behind supports_preview_metadata with no fallback branch.
  * extra_data.preview_method enables previews per-prompt (execution.py:676),
    so ComfyUI does NOT need restarting with --preview-method.
  * We supply our own prompt_id (server.py:883 accepts it), making correlation
    trivial.
  * Both "progress" (main.py:300) and "progress_state" (progress.py:186) exist;
    we accept either.
  * Binary frames: 4-byte BE event id, then payload. Event 4 is
    PREVIEW_IMAGE_WITH_METADATA (4-byte BE metadata length, metadata JSON,
    then JPEG); event 1 is a bare preview.

Timeout is 600 s, not game-poc's 120 s: a cold Klein 9B load measured 138 s,
so the old default failed on the first image of every session.
"""

from __future__ import annotations

import asyncio
import copy
import json
import struct
import time
import uuid
from dataclasses import dataclass, field

import aiohttp

COMFY_HTTP = "http://127.0.0.1:8188"
COMFY_WS = "ws://127.0.0.1:8188/ws"

PREVIEW_IMAGE = 1
PREVIEW_IMAGE_WITH_METADATA = 4


@dataclass
class RunResult:
    prompt_id: str
    ok: bool
    wall_s: float = 0.0
    t_first_progress_s: float | None = None   # = model load + text encode
    t_last_progress_s: float | None = None
    steps_seen: int = 0          # raw frame count (progress + progress_state)
    max_steps: int = 0
    first_step: int | None = None   # sampler step index at t_first_progress_s
    last_step: int = 0              # sampler step index at t_last_progress_s
    previews_seen: int = 0
    images: list[dict] = field(default_factory=list)
    error: str | None = None

    @property
    def sample_s(self) -> float | None:
        """Pure sampling time, excluding load/encode and VAE decode."""
        if self.t_first_progress_s is None or self.t_last_progress_s is None:
            return None
        return self.t_last_progress_s - self.t_first_progress_s

    @property
    def s_per_it(self) -> float | None:
        """Seconds per sampler step.

        Derived from the sampler's own step *values*, not from frame counts:
        ComfyUI emits BOTH "progress" and "progress_state" for each step, so
        counting frames double-counts and halves this figure.
        """
        s = self.sample_s
        if s is None or self.first_step is None:
            return None
        advanced = self.last_step - self.first_step
        if advanced < 1:
            return None
        return s / advanced


def klein_workflow(
    *,
    unet: str,
    clip: str,
    prompt: str,
    seed: int,
    steps: int = 20,
    width: int = 1024,
    height: int = 1024,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    vae: str = "flux2-vae.safetensors",
    negative: str = "",
) -> dict:
    """API-format FLUX.2 Klein workflow.

    NOTE the encoder pairing rule: Klein 9B needs Qwen3-8B, Klein 4B needs
    Qwen3-4B. Mismatching passes every loader and then fails at KSampler with
    'mat1 and mat2 shapes cannot be multiplied'.

    cfg stays at 1.0 — FLUX uses embedded guidance; raising it degrades output.
    """
    return {
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": unet}},
        "2": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": clip, "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": prompt, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode",
              "inputs": {"text": negative, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyFlux2LatentImage",
              "inputs": {"width": width, "height": height, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {
            "seed": seed, "steps": steps, "cfg": cfg,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
            "latent_image": ["6", 0]}},
        "8": {"class_type": "VAEDecode",
              "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "bench", "images": ["8", 0]}},
    }


def _pct_from_progress_state(data: dict) -> tuple[int, int] | None:
    """progress_state carries every node; the sampler is the one that moves."""
    best = None
    for node in data.get("nodes", {}).values():
        mx = node.get("max") or 0
        if mx > 1:  # ignore 0/1-step bookkeeping nodes
            val = node.get("value") or 0
            if best is None or mx > best[1]:
                best = (int(val), int(mx))
    return best


async def run_once(
    workflow: dict,
    *,
    previews: bool = True,
    timeout: float = 600.0,
    prompt_id: str | None = None,
) -> RunResult:
    prompt_id = prompt_id or str(uuid.uuid4())
    client_id = str(uuid.uuid4())
    res = RunResult(prompt_id=prompt_id, ok=False)

    body = {
        "prompt": copy.deepcopy(workflow),
        "client_id": client_id,
        "prompt_id": prompt_id,
    }
    if previews:
        body["extra_data"] = {"preview_method": "latent2rgb"}

    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as http:
            # WS first — a cached job can complete before we would otherwise listen.
            async with http.ws_connect(f"{COMFY_WS}?clientId={client_id}") as ws:
                # MUST be the first frame, or zero previews arrive.
                await ws.send_json({
                    "type": "feature_flags",
                    "data": {"supports_preview_metadata": True},
                })

                async with http.post(f"{COMFY_HTTP}/prompt", json=body) as r:
                    if r.status != 200:
                        res.error = f"submit {r.status}: {(await r.text())[:300]}"
                        return res

                async with asyncio.timeout(timeout):
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            m = json.loads(msg.data)
                            mt, d = m.get("type"), m.get("data", {})
                            if d.get("prompt_id") not in (None, prompt_id):
                                continue

                            if mt in ("progress_state", "progress"):
                                if mt == "progress_state":
                                    pr = _pct_from_progress_state(d)
                                else:
                                    mx = int(d.get("max") or 0)
                                    pr = (int(d.get("value") or 0), mx) if mx > 1 else None
                                if pr:
                                    val, mx = pr
                                    now = time.time() - t0
                                    if res.t_first_progress_s is None:
                                        res.t_first_progress_s = now
                                        res.first_step = val
                                    res.t_last_progress_s = now
                                    res.last_step = max(res.last_step, val)
                                    res.steps_seen += 1
                                    res.max_steps = max(res.max_steps, mx)
                            elif mt == "executed":
                                imgs = d.get("output", {}).get("images", [])
                                if imgs:
                                    res.images.extend(imgs)
                            elif mt == "execution_error":
                                res.error = str(d.get("exception_message"))[:400]
                                return res
                            elif mt == "execution_success":
                                res.ok = True
                                res.wall_s = time.time() - t0
                                if not res.images:
                                    res.images = await _from_history(http, prompt_id)
                                return res
                            elif mt == "executing" and d.get("node") is None:
                                res.ok = True
                                res.wall_s = time.time() - t0
                                if not res.images:
                                    res.images = await _from_history(http, prompt_id)
                                return res

                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            ev = struct.unpack(">I", msg.data[:4])[0]
                            if ev in (PREVIEW_IMAGE, PREVIEW_IMAGE_WITH_METADATA):
                                res.previews_seen += 1

                        elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                          aiohttp.WSMsgType.ERROR):
                            res.error = "websocket closed"
                            return res
    except asyncio.TimeoutError:
        res.error = f"timeout after {timeout}s"
    except Exception as e:  # noqa: BLE001 - benchmark harness
        res.error = f"{type(e).__name__}: {e}"

    res.wall_s = time.time() - t0
    return res


async def _from_history(http: aiohttp.ClientSession, prompt_id: str) -> list[dict]:
    for _ in range(5):
        async with http.get(f"{COMFY_HTTP}/history/{prompt_id}") as r:
            hist = await r.json()
        if prompt_id in hist:
            out = []
            for node_out in hist[prompt_id].get("outputs", {}).values():
                out.extend(node_out.get("images", []))
            return out
        await asyncio.sleep(0.5)
    return []


async def free_vram(*, unload_models: bool = True, free_memory: bool = True,
                    wait_s: float = 25.0) -> float:
    """POST /free and wait for torch VRAM to actually come back.

    /free sets a queue flag and notifies the worker, so it takes effect even
    when idle — but the GC step is rate-limited to once per 10 s, so poll
    rather than assuming. Returns seconds until VRAM rose, or -1 on timeout.
    """
    async with aiohttp.ClientSession() as http:
        async with http.get(f"{COMFY_HTTP}/system_stats") as r:
            before = (await r.json())["devices"][0]["vram_free"]
        await http.post(f"{COMFY_HTTP}/free",
                        json={"unload_models": unload_models,
                              "free_memory": free_memory})
        t0 = time.time()
        while time.time() - t0 < wait_s:
            await asyncio.sleep(0.25)
            async with http.get(f"{COMFY_HTTP}/system_stats") as r:
                now = (await r.json())["devices"][0]["vram_free"]
            if now > before * 1.20:  # meaningfully more free than we started with
                return time.time() - t0
    return -1.0
