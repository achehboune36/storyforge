"""The single owner of GPU residency.

No other module may call ComfyUI's /free or Ollama's keep_alive. Everything
goes through a lease:

    async with broker.lease(Consumer.LLM, session_id):
        ...stream the turn...

    async with broker.lease(Consumer.IMAGE, session_id):
        ...generate one image...

## Why "asymmetric"

Phase 0 measured the two handoffs as wildly different:

    Ollama evict    0.30 s      ComfyUI /free      1.0 s
    Ollama load     ~4 s        ComfyUI reload    ~25 s

So the plan's binary COEXIST/SWAP framing was wrong. Cycling the cheap side
costs ~4 s a turn; cycling the expensive side costs ~30 s. ComfyUI therefore
stays resident for the life of the process, and only the LLM comes and goes.

Full co-residency was measured too (E7): it works — 11827 of 12288 MiB — but
leaves 461 MiB of margin and costs the LLM 60% of its throughput while an image
samples. Not worth it to save 4 s.

## Headroom is checked, not assumed

ComfyUI does **not** release VRAM when idle; it holds 9519 MiB against an
11007 MiB working peak. Loading the LLM on top is the tightest moment in the
whole system, so the broker measures free VRAM first and frees ComfyUI if the
margin is too thin. That path costs the ~25 s reload, which is why it is a
fallback and not the design.

## Acting mid-generation

The user is never blocked from typing — that is the product premise. When a
turn needs the GPU while an image is still sampling, the broker does not
blindly kill it: 20 s of work is worth more than 5 s of latency. It estimates
the remaining time from ComfyUI's own progress and either waits or preempts.
Preemption is safe because seed and prompt are persisted and Phase 1 proved a
re-run reproduces pixel-identical output.
"""

from __future__ import annotations

import asyncio
import contextlib
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from backend.comfy_client import ComfyClient
from backend.llm_ollama import OllamaClient


class Consumer(Enum):
    LLM = "llm"
    IMAGE = "image"


class State(str, Enum):
    IDLE = "idle"
    LLM_RESIDENT = "llm_resident"
    IMAGE_RUNNING = "image_running"
    EVICTING_LLM = "evicting_llm"
    FREEING_COMFY = "freeing_comfy"


def gpu_used_mib() -> int:
    """Whole-device usage. nvidia-smi in WSL sees the Windows desktop too,
    which is exactly right — that memory is genuinely unavailable to us."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


@dataclass
class BrokerStats:
    llm_loads: int = 0
    llm_evicts: int = 0
    comfy_frees: int = 0
    preemptions: int = 0
    waits_for_image: int = 0
    low_headroom_events: int = 0
    last_headroom_mib: int = 0


class GpuBroker:
    def __init__(self, cfg: dict, comfy: ComfyClient, llm: OllamaClient):
        g = cfg["gpu"]
        self.total_mib = int(g["device_total_mib"])
        self.min_headroom = int(g["min_headroom_mib"])
        self.idle_evict_s = float(g.get("idle_evict_s", 300))
        self.evict_wait_s = float(g.get("evict_wait_timeout_s", 15))
        self.free_wait_s = float(g.get("free_wait_timeout_s", 25))
        self.max_preempt_wait_s = float(g.get("max_preempt_wait_s", 25))

        self.comfy = comfy
        self.llm = llm

        self._lock = asyncio.Lock()          # one GPU phase at a time
        self._image_sem = asyncio.Semaphore(1)
        self._state = State.IDLE
        self._llm_loaded = False
        self._last_use = time.time()
        self._llm_wanted = asyncio.Event()   # set when a turn is waiting
        self.stats = BrokerStats()

        # Set by the job runner so the broker can ask an in-flight image how
        # far along it is, and cancel it if waiting would cost too much.
        self.image_progress_hint: Callable[[], tuple[float, str] | None] | None = None
        self.image_cancel: Callable[[], asyncio.Future] | None = None

    # ------------------------------------------------------------------ state

    @property
    def state(self) -> State:
        return self._state

    def snapshot(self) -> dict:
        used = gpu_used_mib()
        return {
            "state": self._state.value,
            "llm_loaded": self._llm_loaded,
            "vram_used_mib": used,
            "vram_free_mib": (self.total_mib - used) if used >= 0 else None,
            "idle_s": round(time.time() - self._last_use, 1),
            "stats": self.stats.__dict__,
        }

    # ----------------------------------------------------------------- leases

    @contextlib.asynccontextmanager
    async def lease(self, consumer: Consumer, owner: str = ""):
        if consumer is Consumer.LLM:
            async with self._llm_lease(owner):
                yield
        else:
            async with self._image_lease(owner):
                yield

    @contextlib.asynccontextmanager
    async def _llm_lease(self, owner: str):
        self._llm_wanted.set()
        try:
            await self._yield_gpu_from_image()
            async with self._lock:
                await self._ensure_llm_resident()
                self._state = State.LLM_RESIDENT
                self._last_use = time.time()
                try:
                    yield
                finally:
                    self._last_use = time.time()
        finally:
            self._llm_wanted.clear()

    @contextlib.asynccontextmanager
    async def _image_lease(self, owner: str):
        # Serialise images even though ComfyUI has its own queue: we want our
        # own visibility, ETA and cancel handle.
        async with self._image_sem:
            async with self._lock:
                await self._evict_llm()
                self._state = State.IMAGE_RUNNING
                self._last_use = time.time()
                try:
                    yield
                finally:
                    self._state = State.IDLE
                    self._last_use = time.time()

    # -------------------------------------------------------------- residency

    @property
    def _llm_is_remote(self) -> bool:
        """A cloud narrator occupies no local VRAM, so no handoff is needed."""
        return getattr(self.llm, "host", "").startswith("http") and \
            "127.0.0.1" not in getattr(self.llm, "host", "")

    async def _ensure_llm_resident(self) -> None:
        """Load the LLM, freeing ComfyUI first only if headroom demands it."""
        if self._llm_is_remote:
            self._llm_loaded = False
            return
        used = gpu_used_mib()
        headroom = (self.total_mib - used) if used >= 0 else self.total_mib
        self.stats.last_headroom_mib = headroom

        if headroom < self.min_headroom:
            # The tightest moment in the system. Falling back here costs the
            # ~25 s ComfyUI reload, so it is worth logging loudly.
            self.stats.low_headroom_events += 1
            self._state = State.FREEING_COMFY
            await self.comfy.free(unload_models=True, free_memory=True,
                                  wait_s=self.free_wait_s)
            self.stats.comfy_frees += 1

        if not self._llm_loaded:
            self.stats.llm_loads += 1
            self._llm_loaded = True   # optimistic; the stream itself loads it

    async def _evict_llm(self) -> None:
        if self._llm_is_remote or not self._llm_loaded:
            return
        self._state = State.EVICTING_LLM
        before = gpu_used_mib()
        await self.llm.evict()
        self.stats.llm_evicts += 1
        self._llm_loaded = False
        # Measured at ~0.30 s, but poll rather than trust it.
        t0 = time.time()
        while time.time() - t0 < self.evict_wait_s:
            await asyncio.sleep(0.1)
            now = gpu_used_mib()
            if now < 0 or now < before - 400:
                return

    # ------------------------------------------------- preemption / arbitration

    async def _yield_gpu_from_image(self) -> None:
        """A turn wants the GPU. Decide whether to wait out the current image.

        Waiting is usually right: an image 3 s from done is worth more finished
        than restarted. But a 20 s wait with the user staring at a prompt box
        is not, so past the threshold we interrupt — safe only because the job
        is deterministic and re-enqueued.
        """
        if self._state is not State.IMAGE_RUNNING:
            return
        hint = self.image_progress_hint() if self.image_progress_hint else None
        remaining = hint[0] if hint else None

        if remaining is not None and remaining <= self.max_preempt_wait_s:
            self.stats.waits_for_image += 1
            return   # the lock will be released when the image finishes

        if self.image_cancel is not None:
            self.stats.preemptions += 1
            with contextlib.suppress(Exception):
                await self.image_cancel()

    # ------------------------------------------------------------- lifecycle

    async def reconcile(self) -> None:
        """On boot, nothing is trusted: evict the LLM, keep ComfyUI as found."""
        with contextlib.suppress(Exception):
            await self.llm.evict()
        self._llm_loaded = False
        self._state = State.IDLE
        self._last_use = time.time()

    async def idle_watchdog(self, poll_s: float = 30.0) -> None:
        """Evict the LLM after a quiet period.

        A forgotten browser tab must not squat on 2.5 GB. ComfyUI is left
        alone deliberately — reloading it costs 25 s, and it is the resident
        side of the asymmetric design.
        """
        while True:
            await asyncio.sleep(poll_s)
            if (self._llm_loaded and self._state is State.IDLE
                    and time.time() - self._last_use > self.idle_evict_s):
                async with self._lock:
                    await self._evict_llm()
                    self._state = State.IDLE

    async def foreign_jobs(self) -> int:
        """Count queue entries we did not submit.

        Someone opening ComfyUI's own web UI would otherwise have their work
        silently unloaded by our /free.
        """
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{self.comfy.http_url}/queue") as r:
                    q = await r.json()
            return len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        except Exception:
            return 0
