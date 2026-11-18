"""E6 + E7 — Ollama characterisation and the coexistence crux.

E6 measures, per model: cold load -> first token, decode tok/s, resident VRAM
at two context sizes, and eviction latency. That last one is currently a pure
assumption in the plan — `keep_alive: 0` is documented to evict immediately,
but "immediately" has never been timed on this machine, and the SWAP strategy
pays that cost twice per story turn.

E7 pins an LLM and Klein 4B simultaneously and interleaves them. It decides
COEXIST vs SWAP. The failure mode to watch for is the driver silently spilling
to host RAM: nvidia-smi pins near the device total while s/it jumps a cliff.
On WSL this is silent, so it is detected here rather than assumed away.

Run with ComfyUI's venv (has aiohttp):
    /home/one/ComfyUI/venv/bin/python -m bench.ollama_bench e6
    /home/one/ComfyUI/venv/bin/python -m bench.ollama_bench e7
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.comfy_bench import klein_workflow, run_once  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
STORY_PROMPT = (
    "You are narrating a noir detective story in second person. "
    "The player has just pushed open the door of a rain-soaked jazz bar. "
    "Write 4 vivid paragraphs describing what they find inside."
)


def _desc(r) -> str:
    """Human summary that survives a None s_per_it (cache hit / no progress)."""
    if not r.ok:
        return f"FAILED: {r.error}"
    spi = f"{r.s_per_it:.3f}" if r.s_per_it is not None else "n/a (cached?)"
    return f"{r.wall_s:.1f}s  s/it={spi}  steps={r.steps_seen}"


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return -1


async def stream_chat(model: str, prompt: str, *, num_ctx: int = 8192,
                      keep_alive: str | int = "5m",
                      max_tokens: int = 300) -> dict:
    """Stream one completion. Returns timing + token counts."""
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "keep_alive": keep_alive,
        "options": {"num_ctx": num_ctx, "num_predict": max_tokens},
    }
    t0 = time.time()
    t_first = None
    ntok = 0
    text_len = 0
    async with aiohttp.ClientSession() as s:
        async with s.post(f"{OLLAMA}/api/chat", json=body,
                          timeout=aiohttp.ClientTimeout(total=600)) as r:
            async for raw in r.content:
                if not raw.strip():
                    continue
                try:
                    m = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                chunk = m.get("message", {}).get("content", "")
                if chunk:
                    if t_first is None:
                        t_first = time.time() - t0
                    ntok += 1
                    text_len += len(chunk)
                if m.get("done"):
                    break
    total = time.time() - t0
    decode = total - (t_first or 0)
    return {
        "t_first_token_s": t_first,
        "total_s": total,
        "chunks": ntok,
        "chars": text_len,
        "tok_per_s": (ntok / decode) if decode > 0 else 0.0,
    }


async def evict(model: str) -> float:
    """keep_alive:0 and time how long VRAM actually takes to come back."""
    before = gpu_used_mib()
    async with aiohttp.ClientSession() as s:
        await s.post(f"{OLLAMA}/api/chat", json={
            "model": model, "messages": [], "keep_alive": 0, "stream": False},
            timeout=aiohttp.ClientTimeout(total=60))
    t0 = time.time()
    while time.time() - t0 < 30:
        await asyncio.sleep(0.25)
        now = gpu_used_mib()
        if now >= 0 and now < before - 400:   # meaningfully released
            return time.time() - t0
    return -1.0


async def exp_e6() -> int:
    print("E6 — Ollama characterisation\n")
    for model in ("llama3.2:3b", "llama3.1:8b"):
        print(f"--- {model}")
        await evict(model)
        await asyncio.sleep(2)
        base = gpu_used_mib()

        for num_ctx in (4096, 8192):
            await evict(model)
            await asyncio.sleep(2)
            base = gpu_used_mib()
            r = await stream_chat(model, STORY_PROMPT, num_ctx=num_ctx,
                                  keep_alive="5m", max_tokens=300)
            resident = gpu_used_mib() - base
            print(f"  num_ctx={num_ctx:<5} first_token={r['t_first_token_s']:.2f}s  "
                  f"{r['tok_per_s']:.1f} chunk/s  {r['chars']} chars  "
                  f"resident~{resident} MiB")

            # warm decode, model already loaded
            r2 = await stream_chat(model, STORY_PROMPT, num_ctx=num_ctx,
                                   keep_alive="5m", max_tokens=300)
            print(f"                warm  first_token={r2['t_first_token_s']:.2f}s  "
                  f"{r2['tok_per_s']:.1f} chunk/s")

        t_evict = await evict(model)
        print(f"  eviction: {t_evict:.2f}s" if t_evict >= 0 else
              "  eviction: TIMED OUT (vram never dropped)")
        print()
    return 0


async def exp_e7(llm: str = "llama3.2:3b", width: int = 1024, height: int = 1024,
                 unet: str = "flux-2-klein-4b-Q8_0.gguf", steps: int = 20) -> int:
    """The crux: both pinned, running at once."""
    print(f"E7 — coexistence: {llm} + {unet} @ {width}x{height}, {steps} steps\n")

    def wf_for(seed: int) -> dict:
        # Seeds MUST differ between runs: an identical workflow is served from
        # ComfyUI's cache instantly, which silently fakes a 0s generation.
        return klein_workflow(
            unet=unet, clip="Qwen3-4B-Q6_K.gguf",
            prompt="a rain-soaked jazz bar interior, amber light, cigarette haze, cinematic",
            seed=seed, steps=steps, width=width, height=height)

    # --- solo baselines -------------------------------------------------
    await evict(llm)
    await asyncio.sleep(2)
    print("solo image...")
    img_solo = await run_once(wf_for(4242))
    print("  image solo: " + _desc(img_solo))

    print("solo llm (pinned)...")
    llm_solo = await stream_chat(llm, STORY_PROMPT, keep_alive=-1, max_tokens=300)
    print(f"  llm solo: {llm_solo['tok_per_s']:.1f} chunk/s  "
          f"first_token={llm_solo['t_first_token_s']:.2f}s")
    pinned = gpu_used_mib()
    print(f"  vram with llm pinned: {pinned} MiB")

    # --- concurrent ------------------------------------------------------
    print("\nconcurrent (llm pinned, image running)...")
    peak = {"v": 0}

    async def watch():
        while True:
            peak["v"] = max(peak["v"], gpu_used_mib())
            await asyncio.sleep(0.25)

    w = asyncio.create_task(watch())
    t0 = time.time()
    img_task = asyncio.create_task(run_once(wf_for(9137)))
    await asyncio.sleep(1.0)  # let the image job get going
    llm_conc = await stream_chat(llm, STORY_PROMPT, keep_alive=-1, max_tokens=300)
    img_conc = await img_task
    w.cancel()

    print("  image concurrent: " + _desc(img_conc))
    print(f"  llm concurrent:   {llm_conc['tok_per_s']:.1f} chunk/s")
    print(f"  PEAK VRAM:        {peak['v']} MiB of 12288")
    print(f"  wall for both:    {time.time() - t0:.1f}s")

    # --- verdict ---------------------------------------------------------
    B = 10.7 * 1024
    if img_conc.ok and img_solo.ok and img_solo.s_per_it and img_conc.s_per_it:
        deg_img = img_conc.s_per_it / img_solo.s_per_it - 1
        deg_llm = 1 - (llm_conc["tok_per_s"] / llm_solo["tok_per_s"]) if llm_solo["tok_per_s"] else 1
        print(f"\n  image degradation: {deg_img*100:+.1f}%")
        print(f"  llm degradation:   {deg_llm*100:+.1f}%")
        ok = peak["v"] <= B and deg_img <= 0.25 and llm_conc["tok_per_s"] >= 15
        print(f"\n  VERDICT: {'COEXIST' if ok else 'SWAP'}")
        print(f"    peak {peak['v']} <= {B:.0f}? {peak['v'] <= B}")
        print(f"    image degradation <= 25%? {deg_img <= 0.25}")
        print(f"    llm >= 15 chunk/s? {llm_conc['tok_per_s'] >= 15}")
    else:
        print("\n  VERDICT: SWAP (a concurrent run failed outright)")

    await evict(llm)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e6", "e7"])
    ap.add_argument("--llm", default="llama3.2:3b")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--unet", default="flux-2-klein-4b-Q8_0.gguf")
    ap.add_argument("--steps", type=int, default=20)
    a = ap.parse_args()
    if a.experiment == "e6":
        return asyncio.run(exp_e6())
    return asyncio.run(exp_e7(a.llm, a.width, a.height, a.unet, a.steps))


if __name__ == "__main__":
    raise SystemExit(main())
