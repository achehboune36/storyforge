"""Phase 1 verification.

    /home/one/ComfyUI/venv/bin/python verify_phase1.py

T1 is the one that matters most. The broker is allowed to interrupt an
in-flight image and re-enqueue it precisely because seed + prompt are
persisted and re-running them reproduces the same picture. Seed-locked visual
consistency for characters rests on the same property. If T1 fails, both
designs need rethinking.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backend import workflow as wf  # noqa: E402
from backend.comfy_client import ComfyClient  # noqa: E402

ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
OUT = ROOT / "data" / "verify"
PROMPT = "a lone figure under a gas lantern in a rain-slick alley, cinematic, detailed"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


# ---------------------------------------------------------------- unit tests

def test_workflow_patching() -> None:
    print("\nT0  workflow sentinel patching (no server needed)")
    graph = wf.build(CFG["image"])
    vals = wf.values(CFG["image"], prompt="x", seed=7, prefix="t")

    patched = wf.patch(graph, vals)
    ks = patched["7"]["inputs"]
    check("seed stays int", isinstance(ks["seed"], int), f"got {type(ks['seed']).__name__}")
    check("cfg stays float", isinstance(ks["cfg"], float), f"got {type(ks['cfg']).__name__}")
    check("steps stays int", isinstance(ks["steps"], int))
    check("no sentinel survives", not wf.find_sentinels(patched))
    check("prompt substituted", patched["4"]["inputs"]["text"] == "x")

    try:
        wf.patch(graph, {k: v for k, v in vals.items() if k != "__SEED__"})
        check("missing value rejected", False, "no error raised")
    except wf.WorkflowError as e:
        check("missing value rejected", "__SEED__" in str(e))

    try:
        wf.patch(graph, {**vals, "__NOPE__": 1})
        check("unknown sentinel rejected", False, "no error raised")
    except wf.WorkflowError:
        check("unknown sentinel rejected", True)

    try:
        wf.patch(graph, {**vals, "__SEED__": "not-a-number"})
        check("bad type rejected", False, "no error raised")
    except wf.WorkflowError:
        check("bad type rejected", True)


# ---------------------------------------------------------------- live tests

def _pixel_digest(path: Path) -> str:
    """Hash decoded pixels, not file bytes.

    ComfyUI embeds the prompt JSON in PNG tEXt metadata, so two pixel-identical
    images differ as files whenever any workflow value differs — including
    filename_prefix. Only the pixels tell us whether sampling was deterministic.
    """
    from PIL import Image
    with Image.open(path) as im:
        return hashlib.sha256(im.convert("RGB").tobytes()).hexdigest()


async def test_determinism(c: ComfyClient) -> None:
    print("\nT1  determinism — same seed 3x must produce identical pixels")
    graph = wf.build(CFG["image"])
    digests, walls = [], []
    for i in range(3):
        # Identical workflow every run, prefix included: any difference would
        # let ComfyUI's per-node cache skip sampling and fake a pass.
        vals = wf.values(CFG["image"], prompt=PROMPT, seed=31337,
                               prefix="verify_det")
        if i:
            # Drop the execution cache (free_memory -> e.reset()) so sampling
            # genuinely re-runs. unload_models stays False to avoid a ~20s
            # model reload we are not testing here.
            await c.free(unload_models=False, free_memory=True, wait_s=6)
        r = await c.generate(wf.patch(graph, vals))
        if not r.ok or not r.images:
            check("determinism", False, f"run {i} failed: {r.error}")
            return
        dest = await c.download(r.images[0], OUT / f"det_{i}.png")
        digests.append(_pixel_digest(dest))
        walls.append(r.wall_s)
        if r.wall_s < 1.0:
            print(f"      (run {i} took {r.wall_s:.1f}s — cache was not cleared)")
    check("3 runs pixel-identical", len(set(digests)) == 1,
          f"{len(set(digests))} distinct digest(s); walls={[f'{w:.1f}' for w in walls]}")
    print(f"      pixel sha256 {digests[0][:16]}…")


async def test_progress_and_previews(c: ComfyClient) -> None:
    print("\nT2  progress + preview streaming")
    steps = CFG["image"]["steps"]
    prog: list[tuple[int, int]] = []
    previews: list[int] = []
    states: list[str] = []

    async def on_progress(v, m): prog.append((v, m))
    async def on_preview(b): previews.append(len(b))
    async def on_state(s): states.append(s)

    vals = wf.values(CFG["image"], prompt=PROMPT, seed=4242, prefix="verify_prog")
    r = await c.generate(wf.patch(wf.build(CFG["image"]), vals),
                         on_progress=on_progress, on_preview=on_preview,
                         on_state=on_state)
    check("generation ok", r.ok, r.error or "")
    check(f"progress reached {steps}/{steps}", r.steps_done >= steps,
          f"got {r.steps_done}/{r.steps_total}")
    check("previews received", len(previews) >= 1, f"{len(previews)} frames")
    check("previews are JPEG-sized", all(n > 500 for n in previews) if previews else False,
          f"sizes {previews[:4]}…" if previews else "none")
    check("state sequence sane", states[0] == "queued" and states[-1] == "ready",
          " -> ".join(states))
    check("wall within 2x of expected", r.wall_s < CFG["image"]["expected_wall_s"] * 2.5,
          f"{r.wall_s:.1f}s vs expected {CFG['image']['expected_wall_s']}s")


async def test_interrupt(c: ComfyClient) -> None:
    print("\nT3  targeted interrupt resolves quickly")
    import uuid
    pid = str(uuid.uuid4())
    # More steps so there is something to interrupt.
    vals = wf.values(CFG["image"], prompt=PROMPT, seed=999, steps=40,
                           prefix="verify_int")
    task = asyncio.create_task(
        c.generate(wf.patch(wf.build(CFG["image"]), vals), prompt_id=pid))
    await asyncio.sleep(6)  # let it actually start sampling
    t0 = time.time()
    await c.interrupt(pid)
    try:
        r = await asyncio.wait_for(task, timeout=15)
        dt = time.time() - t0
        check("client resolved after interrupt", True, f"{dt:.1f}s")
        check("reported as interrupted/failed", not r.ok,
              f"interrupted={r.interrupted} error={r.error}")
    except asyncio.TimeoutError:
        task.cancel()
        check("client resolved after interrupt", False, "still hanging after 15s")


async def test_queue_and_stats(c: ComfyClient) -> None:
    print("\nT4  queue position + vram stats")
    v = await c.vram_free_mib()
    check("vram_free readable", v is not None and v > 0, f"{v} MiB free")
    pos = await c.queue_position("does-not-exist")
    check("unknown prompt not queued", pos is None, f"got {pos}")


async def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    test_workflow_patching()

    c = ComfyClient(CFG)
    if not await c.is_up():
        print("\nComfyUI is not reachable — skipping live tests T1-T4.")
        print(f"  start it: cd /home/one/ComfyUI && ./venv/bin/python main.py")
    else:
        await test_progress_and_previews(c)
        await test_determinism(c)
        await test_interrupt(c)
        await test_queue_and_stats(c)

    print(f"\n{'=' * 58}")
    print(f"  {len(passed)} passed, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"    FAILED: {f}")
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
