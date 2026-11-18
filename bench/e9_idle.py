"""E9 — ComfyUI idle footprint, and whether the asymmetric handoff works.

E7 showed LLM + ComfyUI *both working* peaks at 11827/12288 MiB: it fits, but
with 461 MiB spare and a 60% LLM slowdown. E6 showed the two handoffs cost
wildly different amounts (Ollama evict 0.3s / load ~4s; ComfyUI reload ~25s).

That suggests a third strategy the plan's binary COEXIST/SWAP framing missed:
keep ComfyUI resident permanently, cycle only the LLM. Its viability rests on
one unmeasured number — what ComfyUI holds *between* generations, when its
sampling activations are freed but its weights are not.

This measures the four states that decide it:
    A  baseline        ComfyUI up, no model loaded
    B  peak            during a generation
    C  idle-resident   after the generation settles
    D  C + LLM loaded  the asymmetric design's real worst case

Verdict is D against the device total.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.comfy_bench import klein_workflow, run_once  # noqa: E402
from bench.ollama_bench import evict, gpu_used_mib, stream_chat  # noqa: E402

TOTAL = 12288
LLM = "llama3.2:3b"


async def sample_peak(coro):
    """Run coro while polling VRAM at 4 Hz; return (result, peak)."""
    peak = {"v": 0}

    async def watch():
        while True:
            peak["v"] = max(peak["v"], gpu_used_mib())
            await asyncio.sleep(0.25)

    w = asyncio.create_task(watch())
    try:
        res = await coro
    finally:
        w.cancel()
    return res, peak["v"]


async def main() -> int:
    print("E9 — idle footprint / asymmetric handoff\n")

    await evict(LLM)
    await asyncio.sleep(2)

    a = gpu_used_mib()
    print(f"A  baseline (ComfyUI up, nothing loaded):  {a:>6} MiB")

    wf = klein_workflow(
        unet="flux-2-klein-4b-Q8_0.gguf", clip="Qwen3-4B-Q6_K.gguf",
        prompt="a rain-soaked jazz bar interior, amber light, cigarette haze, cinematic",
        seed=31337, steps=20, width=768, height=768)

    r, b = await sample_peak(run_once(wf))
    if not r.ok:
        print(f"generation failed: {r.error}")
        return 1
    print(f"B  peak during generation:                 {b:>6} MiB   "
          f"({r.wall_s:.1f}s wall)")

    # Let activations settle; weights should stay resident.
    for wait in (3, 8, 15):
        await asyncio.sleep(wait if wait == 3 else wait - 3)
        c = gpu_used_mib()
        print(f"C  idle-resident (+{wait:>2}s after finish):        {c:>6} MiB")

    # Now the asymmetric design's worst case: ComfyUI idle, LLM loaded.
    print(f"\n   loading {LLM} on top of idle ComfyUI...")
    res = await stream_chat(LLM, "Write one vivid sentence about rain.",
                            num_ctx=8192, keep_alive=-1, max_tokens=60)
    await asyncio.sleep(2)
    d = gpu_used_mib()
    print(f"D  idle ComfyUI + LLM resident:            {d:>6} MiB   "
          f"({res['tok_per_s']:.0f} chunk/s)")

    t_ev = await evict(LLM)
    await asyncio.sleep(2)
    e = gpu_used_mib()
    print(f"   after LLM evict ({t_ev:.2f}s):                  {e:>6} MiB")

    # --- verdict ---------------------------------------------------------
    head_d = TOTAL - d
    print("\n" + "=" * 62)
    print(f"  LLM weight while ComfyUI idles: {d - c:>6} MiB")
    print(f"  Headroom in the asymmetric worst case (D): {head_d:>5} MiB")
    print(f"  Headroom when generating with LLM evicted:  {TOTAL - b:>5} MiB")
    if head_d > 800:
        print("\n  ASYMMETRIC HANDOFF IS VIABLE.")
        print("  Keep ComfyUI resident; cycle only the LLM.")
        print(f"  Per-turn GPU overhead ~= LLM load + evict, not ComfyUI's ~25s reload.")
    else:
        print("\n  TOO TIGHT — ComfyUI must be freed during the LLM phase.")
        print("  Per-turn cost reverts to ComfyUI's ~25s reload.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
