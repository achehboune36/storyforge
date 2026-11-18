"""ComfyUI client.

Ported from game-poc's `backend/comfyui_service.py`, keeping its one genuinely
load-bearing idea — open the WebSocket *before* POSTing /prompt, so a cached or
instant job cannot finish before we are listening — and adding what a live
reading UI needs:

  * our own `prompt_id` (server.py:883 accepts it), so correlation is trivial
  * the `feature_flags` handshake, which MUST be the first frame or ComfyUI
    sends zero previews (comfy_execution/progress.py:207-231 gates them behind
    supports_preview_metadata with no fallback branch)
  * `extra_data.preview_method`, which enables previews per-prompt
    (execution.py:676) — ComfyUI does NOT need restarting with --preview-method
  * binary preview frame decoding
  * progress -> percentage, via sampler step *values* (both "progress" and
    "progress_state" fire per step, so counting frames double-counts)
  * queue position, targeted /interrupt, /free, /system_stats

Two corrections to the original worth stating:

  * Timeout is 600 s, not 120 s. A cold Klein 9B load measured 138 s, so the
    old default would have failed on the first image of every session.
  * Jobs are owned by the caller, not by a WebSocket. game-poc closed over the
    browser socket, so an image finishing after a disconnect was simply lost.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import aiohttp

# protocol.py BinaryEventTypes
PREVIEW_IMAGE = 1
UNENCODED_PREVIEW_IMAGE = 2
TEXT = 3
PREVIEW_IMAGE_WITH_METADATA = 4

COMFY_INPUT_DIR = Path(os.environ.get("COMFY_DIR", str(Path.home() / "ComfyUI"))) / "input"

ProgressCb = Callable[[int, int], Awaitable[None]]
PreviewCb = Callable[[bytes], Awaitable[None]]
StateCb = Callable[[str], Awaitable[None]]


@dataclass
class ImageResult:
    prompt_id: str
    ok: bool
    wall_s: float = 0.0
    images: list[dict] = field(default_factory=list)
    steps_done: int = 0
    steps_total: int = 0
    previews: int = 0
    interrupted: bool = False
    error: str | None = None


class ComfyClient:
    def __init__(self, cfg: dict):
        host = cfg["comfy"]["host"].rstrip("/")
        self.http_url = host
        self.ws_url = host.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self.timeout_s = float(cfg["image"].get("timeout_s", 600))
        self.preview_method = cfg["image"].get("preview_method", "latent2rgb")

    # ---------------------------------------------------------------- generate

    async def generate(
        self,
        workflow: dict,
        *,
        prompt_id: str | None = None,
        on_progress: ProgressCb | None = None,
        on_preview: PreviewCb | None = None,
        on_state: StateCb | None = None,
        previews: bool = True,
    ) -> ImageResult:
        """Submit a patched API-format workflow and stream its lifecycle.

        Callbacks are awaited inline, so keep them cheap — push onto a queue
        rather than doing work here, or you will stall the socket read loop and
        drop preview frames.
        """
        prompt_id = prompt_id or str(uuid.uuid4())
        client_id = str(uuid.uuid4())
        res = ImageResult(prompt_id=prompt_id, ok=False)

        body: dict = {"prompt": workflow, "client_id": client_id,
                      "prompt_id": prompt_id}
        if previews:
            body["extra_data"] = {"preview_method": self.preview_method}

        t0 = time.time()
        first_step: int | None = None
        try:
            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(f"{self.ws_url}?clientId={client_id}") as ws:
                    # MUST be the first frame on this socket.
                    await ws.send_json({
                        "type": "feature_flags",
                        "data": {"supports_preview_metadata": True},
                    })

                    async with http.post(f"{self.http_url}/prompt", json=body) as r:
                        if r.status != 200:
                            res.error = f"submit {r.status}: {(await r.text())[:400]}"
                            return res

                    if on_state:
                        await on_state("queued")

                    async with asyncio.timeout(self.timeout_s):
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                m = json.loads(msg.data)
                                mt = m.get("type")
                                d = m.get("data", {}) or {}
                                if d.get("prompt_id") not in (None, prompt_id):
                                    continue

                                if mt in ("progress_state", "progress"):
                                    pr = (_step_from_state(d) if mt == "progress_state"
                                          else _step_from_progress(d))
                                    if pr:
                                        val, mx = pr
                                        if first_step is None:
                                            first_step = val
                                            if on_state:
                                                await on_state("running")
                                        res.steps_done = max(res.steps_done, val)
                                        res.steps_total = max(res.steps_total, mx)
                                        if on_progress:
                                            await on_progress(val, mx)

                                elif mt == "executed":
                                    imgs = d.get("output", {}).get("images", [])
                                    if imgs:
                                        res.images.extend(imgs)

                                elif mt == "execution_error":
                                    res.error = str(d.get("exception_message"))[:500]
                                    if on_state:
                                        await on_state("failed")
                                    return res

                                elif mt == "execution_interrupted":
                                    res.interrupted = True
                                    res.error = "interrupted"
                                    if on_state:
                                        await on_state("interrupted")
                                    return res

                                elif mt == "execution_success" or (
                                        mt == "executing" and d.get("node") is None):
                                    res.ok = True
                                    res.wall_s = time.time() - t0
                                    if not res.images:
                                        res.images = await self._from_history(http, prompt_id)
                                    if on_state:
                                        await on_state("ready")
                                    return res

                            elif msg.type == aiohttp.WSMsgType.BINARY:
                                jpeg = _decode_preview(msg.data)
                                if jpeg is not None:
                                    res.previews += 1
                                    if on_preview:
                                        await on_preview(jpeg)

                            elif msg.type in (aiohttp.WSMsgType.CLOSED,
                                              aiohttp.WSMsgType.ERROR):
                                res.error = "websocket closed"
                                return res

        except asyncio.TimeoutError:
            res.error = f"timeout after {self.timeout_s}s"
        except asyncio.CancelledError:
            # The caller gave up; tell ComfyUI so the GPU is not wasted.
            await self.interrupt(prompt_id)
            raise
        except Exception as e:  # noqa: BLE001
            res.error = f"{type(e).__name__}: {e}"

        res.wall_s = time.time() - t0
        return res

    async def _from_history(self, http: aiohttp.ClientSession,
                            prompt_id: str) -> list[dict]:
        """Cached/instant jobs can finish without an `executed` frame."""
        for _ in range(5):
            async with http.get(f"{self.http_url}/history/{prompt_id}") as r:
                hist = await r.json()
            if prompt_id in hist:
                out: list[dict] = []
                for node_out in hist[prompt_id].get("outputs", {}).values():
                    out.extend(node_out.get("images", []))
                return out
            await asyncio.sleep(0.5)
        return []

    # ------------------------------------------------------------------ assets

    async def download(self, image_info: dict, dest: Path) -> Path:
        params = {
            "filename": image_info["filename"],
            "subfolder": image_info.get("subfolder", ""),
            "type": image_info.get("type", "output"),
        }
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{self.http_url}/view", params=params) as r:
                data = await r.read()
        await asyncio.to_thread(dest.write_bytes, data)
        return dest

    async def stage_input(self, src: Path, name: str) -> Path:
        """Copy a file into ComfyUI's input directory.

        PuLID's LoadImage resolves names against ComfyUI/input, so a reference
        portrait has to exist there as well as in our own storage.
        """
        import shutil
        dest = COMFY_INPUT_DIR / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copyfile, src, dest)
        return dest

    # ------------------------------------------------------------------ control

    async def interrupt(self, prompt_id: str | None = None) -> bool:
        """Cancel a job. Targeted when a prompt_id is given (server.py:939)."""
        payload = {"prompt_id": prompt_id} if prompt_id else {}
        try:
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{self.http_url}/interrupt", json=payload) as r:
                    return r.status == 200
        except Exception:
            return False

    async def queue_position(self, prompt_id: str) -> int | None:
        """0 = currently running, 1+ = pending, None = not queued."""
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(f"{self.http_url}/queue") as r:
                    q = await r.json()
        except Exception:
            return None
        for item in q.get("queue_running", []):
            if len(item) > 1 and item[1] == prompt_id:
                return 0
        for i, item in enumerate(q.get("queue_pending", [])):
            if len(item) > 1 and item[1] == prompt_id:
                return i + 1
        return None

    async def vram_free_mib(self) -> int | None:
        try:
            async with aiohttp.ClientSession() as http:
                async with http.get(f"{self.http_url}/system_stats") as r:
                    d = (await r.json())["devices"][0]
            return d["vram_free"] // 1048576
        except Exception:
            return None

    async def free(self, *, unload_models: bool = True, free_memory: bool = True,
                   wait_s: float = 25.0) -> float:
        """POST /free and wait for VRAM to actually come back.

        /free sets a queue flag and notifies the worker, so it applies even when
        idle — but the GC step is rate-limited to once per 10 s, so poll instead
        of assuming. Returns seconds waited, or -1 on timeout.
        """
        before = await self.vram_free_mib()
        try:
            async with aiohttp.ClientSession() as http:
                await http.post(f"{self.http_url}/free",
                                json={"unload_models": unload_models,
                                      "free_memory": free_memory})
        except Exception:
            return -1.0
        if before is None:
            return -1.0
        t0 = time.time()
        while time.time() - t0 < wait_s:
            await asyncio.sleep(0.25)
            now = await self.vram_free_mib()
            if now is not None and now > before + 256:
                return time.time() - t0
        return -1.0

    async def checkpoints(self) -> list[str]:
        """Checkpoint filenames ComfyUI can load, in its own order.

        Asked of the server rather than read off the disk: ComfyUI knows its
        own extra_model_paths, and a file it cannot see is not a model we can
        offer.
        """
        async with aiohttp.ClientSession() as http:
            async with http.get(f"{self.http_url}/object_info/CheckpointLoaderSimple",
                                timeout=aiohttp.ClientTimeout(total=10)) as r:
                r.raise_for_status()
                info = await r.json()
        node = info.get("CheckpointLoaderSimple", {})
        names = node.get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
        return [n for n in names if isinstance(n, str)]

    async def diffusion_models(self) -> list[str]:
        """Names the UNET loaders can load: Anima and Flux live here, not in
        checkpoints/. GGUF only appears if ComfyUI-GGUF is installed, so a
        missing node is an empty list rather than an error.
        """
        out: list[str] = []
        async with aiohttp.ClientSession() as http:
            for node, field in (("UNETLoader", "unet_name"),
                                ("UnetLoaderGGUF", "unet_name")):
                try:
                    async with http.get(f"{self.http_url}/object_info/{node}",
                                        timeout=aiohttp.ClientTimeout(total=10)) as r:
                        if r.status != 200:
                            continue
                        info = await r.json()
                except Exception:  # noqa: BLE001 — an absent node is not an error
                    continue
                names = (info.get(node, {}).get("input", {})
                         .get("required", {}).get(field, [[]])[0])
                out.extend(n for n in names
                           if isinstance(n, str) and n not in out)
        return out

    async def is_up(self) -> bool:
        return await self.vram_free_mib() is not None


# ----------------------------------------------------------------- frame parsing

def _step_from_state(d: dict) -> tuple[int, int] | None:
    """progress_state carries every node; the sampler is the one that moves."""
    best: tuple[int, int] | None = None
    for node in (d.get("nodes") or {}).values():
        mx = node.get("max") or 0
        if mx > 1:  # skip 0/1-step bookkeeping nodes
            val = int(node.get("value") or 0)
            if best is None or mx > best[1]:
                best = (val, int(mx))
    return best


def _step_from_progress(d: dict) -> tuple[int, int] | None:
    mx = int(d.get("max") or 0)
    return (int(d.get("value") or 0), mx) if mx > 1 else None


def _decode_preview(buf: bytes) -> bytes | None:
    """Return the JPEG bytes from a binary frame, or None if it isn't a preview.

    Layout: 4-byte BE event id, then for PREVIEW_IMAGE_WITH_METADATA a 4-byte BE
    metadata length, the metadata JSON, then the image (server.py:1106-1141).
    """
    if len(buf) < 4:
        return None
    event = struct.unpack(">I", buf[:4])[0]
    if event == PREVIEW_IMAGE_WITH_METADATA:
        if len(buf) < 8:
            return None
        mlen = struct.unpack(">I", buf[4:8])[0]
        return buf[8 + mlen:] or None
    if event in (PREVIEW_IMAGE, UNENCODED_PREVIEW_IMAGE):
        return buf[4:] or None
    return None
