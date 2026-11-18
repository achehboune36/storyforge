"""Phase 0 experiment runner.

Run with ComfyUI's venv (it already has aiohttp):

    /home/one/ComfyUI/venv/bin/python -m bench.run_bench e-1
    /home/one/ComfyUI/venv/bin/python -m bench.run_bench e1
    ...

Each run gets its own vram_sampler subprocess; peak VRAM and swap delta are
read back out of that CSV afterwards. Summary rows append to
bench/results/summary.csv so partial progress survives an interrupted session.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench.comfy_bench import RunResult, free_vram, klein_workflow, run_once  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
SUMMARY = RESULTS / "summary.csv"
PYTHON = sys.executable

# Encoder pairing is a hard constraint: 9B->Qwen3-8B, 4B->Qwen3-4B.
MODELS = {
    "klein9b_q6": ("flux-2-klein-9b-Q6_K.gguf", "Qwen3-8B-Q6_K.gguf"),
    "klein4b_q6": ("flux-2-klein-4b-Q6_K.gguf", "Qwen3-4B-Q6_K.gguf"),
    "klein4b_q8": ("flux-2-klein-4b-Q8_0.gguf", "Qwen3-4B-Q6_K.gguf"),
}

PROMPT = ("a lone figure in a rain-slick alley at night, a single gas lantern "
          "overhead, steam rising from a grate, cinematic, highly detailed")

SUMMARY_FIELDS = [
    "exp", "model", "width", "height", "steps", "seed", "run_idx", "phase",
    "ok", "wall_s", "t_first_progress_s", "sample_s", "s_per_it",
    "steps_seen", "steps_advanced", "previews_seen", "peak_smi_mib", "peak_torch_used_mib",
    "swap_delta_mib", "error",
]


class Sampler:
    """vram_sampler subprocess, scoped to one run."""

    def __init__(self, label: str, path: Path):
        self.path = path
        self._p = subprocess.Popen(
            [PYTHON, "-m", "bench.vram_sampler", "--out", str(path), "--label", label],
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        time.sleep(0.6)  # let it take a baseline sample before the run starts

    def stop(self) -> dict:
        self._p.send_signal(signal.SIGTERM)
        try:
            self._p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._p.kill()
        return self._summarise()

    def _summarise(self) -> dict:
        peak_smi = 0
        peak_torch_used = 0
        swap_first = swap_last = None
        try:
            with self.path.open() as fh:
                for row in csv.DictReader(fh):
                    try:
                        if row["smi_used_mib"]:
                            peak_smi = max(peak_smi, int(row["smi_used_mib"]))
                        if row["torch_free_mib"] and row["torch_total_mib"]:
                            used = int(row["torch_total_mib"]) - int(row["torch_free_mib"])
                            peak_torch_used = max(peak_torch_used, used)
                        sw = int(row["swap_used_mib"])
                        swap_first = sw if swap_first is None else swap_first
                        swap_last = sw
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError:
            pass
        return {
            "peak_smi_mib": peak_smi or "",
            "peak_torch_used_mib": peak_torch_used or "",
            "swap_delta_mib": (swap_last - swap_first) if swap_first is not None else "",
        }


def _append(row: dict) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    new = not SUMMARY.exists()
    with SUMMARY.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SUMMARY_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in SUMMARY_FIELDS})


def _fmt(r: RunResult) -> str:
    if not r.ok:
        return f"FAIL {r.error}"
    spi = f"{r.s_per_it:.3f}" if r.s_per_it else "?"
    tfp = f"{r.t_first_progress_s:.1f}" if r.t_first_progress_s else "?"
    return (f"{r.wall_s:6.1f}s wall | load+encode {tfp}s | {spi} s/it | "
            f"{r.steps_seen} prog | {r.previews_seen} prev")


async def one(exp: str, model: str, *, width: int, height: int, steps: int,
              run_idx: int, phase: str, seed: int = 12345) -> RunResult:
    unet, clip = MODELS[model]
    wf = klein_workflow(unet=unet, clip=clip, prompt=PROMPT, seed=seed,
                        steps=steps, width=width, height=height)
    tag = f"{exp}_{model}_{width}x{height}_s{steps}_{phase}{run_idx}"
    s = Sampler(tag, RESULTS / f"{tag}.csv")
    try:
        r = await run_once(wf)
    finally:
        stats = s.stop()

    print(f"  [{tag}] {_fmt(r)}", flush=True)
    _append({
        "exp": exp, "model": model, "width": width, "height": height,
        "steps": steps, "seed": seed, "run_idx": run_idx, "phase": phase,
        "ok": int(r.ok), "wall_s": f"{r.wall_s:.2f}",
        "t_first_progress_s": f"{r.t_first_progress_s:.2f}" if r.t_first_progress_s else "",
        "sample_s": f"{r.sample_s:.2f}" if r.sample_s else "",
        "s_per_it": f"{r.s_per_it:.4f}" if r.s_per_it else "",
        "steps_seen": r.steps_seen,
        "steps_advanced": (r.last_step - r.first_step) if r.first_step is not None else "",
        "previews_seen": r.previews_seen,
        "error": r.error or "", **stats,
    })
    return r


async def exp_gate() -> int:
    """E-1: does Klein 4B load at all? Everything downstream depends on this."""
    print("E-1: Klein 4B gate (512x512, 8 steps)")
    for model in ("klein4b_q6", "klein4b_q8"):
        unet, clip = MODELS[model]
        if not (Path("/home/one/ComfyUI/models/diffusion_models") / unet).exists():
            print(f"  SKIP {model}: {unet} not downloaded")
            continue
        r = await one("e-1", model, width=512, height=512, steps=8,
                      run_idx=0, phase="gate")
        if not r.ok:
            print(f"\nGATE FAILED for {model}: {r.error}")
            print("The COEXIST branch of the VRAM strategy depends on this.")
            return 1
    print("\nE-1 PASSED — Klein 4B loads and samples.")
    return 0


async def exp_models() -> int:
    """E1 + E2: 9B baseline vs 4B, warm, at two resolutions."""
    for model in ("klein9b_q6", "klein4b_q6", "klein4b_q8"):
        unet, _ = MODELS[model]
        if not (Path("/home/one/ComfyUI/models/diffusion_models") / unet).exists():
            print(f"SKIP {model}: not downloaded")
            continue
        for (w, h) in ((1024, 1024), (768, 1152)):
            print(f"\n{model} @ {w}x{h}, 20 steps")
            # First run is cold (model not resident); label it and keep it —
            # the cold/warm delta is exactly what E5 needs.
            await one("e1e2", model, width=w, height=h, steps=20,
                      run_idx=0, phase="cold")
            for i in range(1, 4):
                await one("e1e2", model, width=w, height=h, steps=20,
                          run_idx=i, phase="warm", seed=12345 + i)
        await free_vram()
    return 0


async def exp_steps(model: str) -> int:
    """E3: step sweep. Same seed throughout so images are visually comparable."""
    print(f"E3: step sweep on {model}")
    await one("e3", model, width=1024, height=1024, steps=20, run_idx=0, phase="warmup")
    for steps in (8, 12, 16, 20):
        await one("e3", model, width=1024, height=1024, steps=steps,
                  run_idx=0, phase="sweep", seed=777)
    return 0


async def exp_res(model: str) -> int:
    """E4: resolution sweep. Cost should track w*h."""
    print(f"E4: resolution sweep on {model}")
    await one("e4", model, width=1024, height=1024, steps=20, run_idx=0, phase="warmup")
    for (w, h) in ((768, 768), (768, 1152), (832, 1216), (1024, 1024), (1344, 768)):
        await one("e4", model, width=w, height=h, steps=20,
                  run_idx=0, phase="sweep", seed=777)
    return 0


async def exp_reload(model: str) -> int:
    """E5: the swap tax. Paid twice per turn in SWAP mode."""
    print(f"E5: unload/reload cost on {model}")
    for i in range(3):
        await one("e5", model, width=1024, height=1024, steps=20,
                  run_idx=i, phase="warm", seed=555)
        t = await free_vram()
        print(f"  free_vram round {i}: {t:.2f}s" if t >= 0 else
              f"  free_vram round {i}: TIMED OUT (vram never rose)")
        _append({"exp": "e5", "model": model, "run_idx": i, "phase": "free",
                 "ok": int(t >= 0), "wall_s": f"{t:.2f}"})
        await one("e5", model, width=1024, height=1024, steps=20,
                  run_idx=i, phase="cold", seed=555)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment", choices=["e-1", "e1e2", "e3", "e4", "e5"])
    ap.add_argument("--model", default="klein4b_q6", choices=list(MODELS))
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    runners = {
        "e-1": exp_gate,
        "e1e2": exp_models,
        "e3": lambda: exp_steps(args.model),
        "e4": lambda: exp_res(args.model),
        "e5": lambda: exp_reload(args.model),
    }
    return asyncio.run(runners[args.experiment]())


if __name__ == "__main__":
    raise SystemExit(main())
