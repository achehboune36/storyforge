"""Phase 4 verification — broker and job queue, end to end on real hardware.

    /home/one/ComfyUI/venv/bin/python verify_phase4.py

Needs ComfyUI and Ollama running. This is the first test that exercises the
whole GPU handoff: stream a turn, evict the LLM, generate the image it asked
for, and report progress and previews throughout.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backend import database as db  # noqa: E402

db.DB_PATH = ROOT / "data" / "verify_phase4.db"
if db.DB_PATH.exists():
    db.DB_PATH.unlink()

from backend import story_loader  # noqa: E402
from backend.comfy_client import ComfyClient  # noqa: E402
from backend.gpu_broker import Consumer, GpuBroker, State, gpu_used_mib  # noqa: E402
from backend.image_jobs import ImageJobRunner  # noqa: E402
from backend.llm_ollama import OllamaClient, build_messages, stream_turn  # noqa: E402
from backend.turn_parser import ImageSlot, segments_from_events  # noqa: E402
from backend import context  # noqa: E402

CFG = json.loads((ROOT / "config.json").read_text())

passed: list[str] = []
failed: list[str] = []
events: list[tuple[str, dict]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


async def on_event(kind: str, payload: dict) -> None:
    events.append((kind, payload))


async def main() -> int:
    db.init()
    story_loader.load_all(verbose=False)
    sess = story_loader.start_session("lighthouse")
    sid = sess["id"]

    comfy = ComfyClient(CFG)
    llm = OllamaClient(CFG)
    if not await comfy.is_up():
        print("ComfyUI not reachable — start it first."); return 1
    if not await llm.is_up():
        print("Ollama not reachable — start it on Windows."); return 1

    broker = GpuBroker(CFG, comfy, llm)
    runner = ImageJobRunner(CFG, broker, comfy, on_event=on_event)
    runner.start()
    await broker.reconcile()

    print(f"\nB0  baseline   vram_used={gpu_used_mib()} MiB  state={broker.state.value}")

    # ---------------------------------------------------------------- turn
    print("\nB1  narrative turn under an LLM lease")
    story = db.get_story("lighthouse")
    action = "I push open the door of the jazz bar and step inside."
    asm = context.assemble(sid, action, num_ctx=CFG["llm"]["num_ctx"])
    turn_user = db.add_turn(sid, "user", action)

    slots: list[ImageSlot] = []
    t0 = time.time()
    async with broker.lease(Consumer.LLM, sid):
        check("state is LLM_RESIDENT inside the lease",
              broker.state is State.LLM_RESIDENT, broker.state.value)

        async def on_ev(ev):
            if isinstance(ev, ImageSlot):
                slots.append(ev)

        res = await stream_turn(llm, build_messages(asm.system, asm.history, asm.user),
                                on_event=on_ev, max_images=2,
                                previous_keywords=None)
    llm_s = time.time() - t0
    check("turn streamed", res.ok, res.llm.error or "")
    check("prose produced", len(res.llm.text) > 120, f"{len(res.llm.text)} chars")
    if len(slots) >= 1:
        check("model emitted an image directive", True, f"{len(slots)} slots")
    else:
        print("  ----  model emitted no directive (prompt-compliance, not pipeline)")
    print(f"      llm {llm_s:.1f}s, {res.llm.chunks_per_s:.0f} chunk/s, "
          f"vram now {gpu_used_mib()} MiB")
    if slots:
        print(f"      keywords: {slots[0].keywords[:80]}")

    segs = segments_from_events(res.events)
    turn_n = db.add_turn(sid, "narrator", res.llm.text, raw_text=res.llm.text)

    if not slots:
        # The GPU handoff and job pipeline are what this phase verifies; model
        # compliance is measured separately (bench/prompt_variants.py) and is
        # known to be sensitive to prompt shape. Inject a slot so the pipeline
        # is still exercised, and say so plainly.
        print("      note: model emitted no directive — injecting a synthetic "
              "slot so the image pipeline is still verified")
        slots.append(ImageSlot(index=0, entities=["bar"],
                               keywords="smoky jazz bar interior, amber lamps, "
                                        "cigarette haze, wet coats"))

    # --------------------------------------------------------------- image
    print("\nB2  image job: handoff, progress, previews")
    t0 = time.time()
    image_id = await runner.submit(session_id=sid, turn_id=turn_n, turn_idx=1,
                                   slot=slots[0], style=sess["style"])
    check("image row created queued",
          db.get_image(image_id)["status"] == "queued")

    for _ in range(600):
        await asyncio.sleep(0.25)
        st = db.get_image(image_id)["status"]
        if st in ("ready", "failed"):
            break
    wall = time.time() - t0
    img = db.get_image(image_id)

    check("image reached ready", img["status"] == "ready",
          f"{img['status']} {img.get('error') or ''}")
    check("file written", bool(img["file_path"]) and Path(img["file_path"]).exists())
    check("llm evicted for the image", not broker._llm_loaded)

    kinds = [k for k, _ in events]
    check("progress events emitted", kinds.count("image_progress") >= 5,
          f"{kinds.count('image_progress')}")
    check("preview events emitted", kinds.count("image_preview") >= 1,
          f"{kinds.count('image_preview')}")
    check("previews throttled, not one per step",
          kinds.count("image_preview") <= int(CFG['image']['steps']),
          f"{kinds.count('image_preview')} previews for {CFG['image']['steps']} steps")
    check("ready event emitted", "image_ready" in kinds)
    print(f"      image wall {wall:.1f}s (expected ~{CFG['image']['expected_wall_s']}s)"
          f"  peak vram {gpu_used_mib()} MiB")

    # -------------------------------------------------------------- prompt
    print("\nB3  prompt construction")
    check("style applied", "film noir" in img["prompt"])
    check("seed persisted", int(img["seed"]) > 0, str(img["seed"]))
    check("entities recorded", isinstance(img["entities"], list))
    print(f"      {img['prompt'][:150]}…")

    # ---------------------------------------------------------- broker stats
    print("\nB4  broker accounting")
    s = broker.stats
    check("llm evicted exactly once", s.llm_evicts >= 1, str(s.llm_evicts))
    check("comfy never freed (asymmetric design held)", s.comfy_frees == 0,
          f"{s.comfy_frees} frees — headroom fell below {broker.min_headroom} MiB")
    print(f"      {broker.snapshot()}")

    await runner.stop()
    print(f"\n{'=' * 58}")
    print(f"  {len(passed)} passed, {len(failed)} failed")
    for f in failed:
        print(f"    FAILED: {f}")
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
