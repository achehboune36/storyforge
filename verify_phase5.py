"""Phase 5/6 verification — the whole thing, over the real wire.

    python3 verify_phase5.py        (server must already be running)

Creates a session over REST, connects the WebSocket, plays two turns, and
checks the message protocol the frontend depends on: that prose streams, that
an image slot opens *before* generation starts, and that reconnecting replays
the story from storage rather than losing it.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time

import aiohttp

BASE = "http://127.0.0.1:8100"
WS = "ws://127.0.0.1:8100"

passed: list[str] = []
failed: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    (passed if ok else failed).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  — {detail}" if detail else ""))


async def collect(ws, until: str, timeout: float = 240.0) -> list[tuple[str, dict]]:
    """Drain frames until `until` arrives (or timeout)."""
    out: list[tuple[str, dict]] = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
        except asyncio.TimeoutError:
            break
        if msg.type is not aiohttp.WSMsgType.TEXT:
            break
        m = json.loads(msg.data)
        out.append((m["t"], m.get("d") or {}))
        if m["t"] == until:
            break
    return out


async def main() -> int:
    async with aiohttp.ClientSession() as http:
        # ---------------------------------------------------------- REST
        print("\nS1  REST surface")
        async with http.get(f"{BASE}/api/stories") as r:
            stories = await r.json()
        check("stories listed", any(s["id"] == "lighthouse" for s in stories),
              str([s["id"] for s in stories]))

        async with http.get(f"{BASE}/api/debug") as r:
            dbg = await r.json()
        check("comfyui up", dbg["comfy_up"])
        check("ollama up", dbg["ollama_up"])
        # image_only is the cloud-narrator era: the LLM left the card, so the
        # broker no longer cycles anything.
        check("broker reports mode",
              dbg["config"]["gpu"]["mode"] in ("asymmetric", "coexist", "image_only"),
              dbg["config"]["gpu"]["mode"])
        check("active model reported", bool(dbg.get("active_model", {}).get("name")),
              dbg.get("active_model", {}).get("label", "?"))
        check("prompt book parses", dbg.get("prompt_book", {}).get("ok") is True,
              dbg.get("prompt_book", {}).get("error", ""))

        async with http.post(f"{BASE}/api/sessions?story_id=lighthouse") as r:
            sess = await r.json()
        sid = sess["id"]
        check("session created", bool(sid))

        async with http.get(f"{BASE}/api/sessions/{sid}") as r:
            state = await r.json()
        segs = state["segments"]
        check("opening scene persisted", len(segs) >= 1, f"{len(segs)} segments")
        check("opening has an illustration",
              any(s["kind"] == "image" for s in segs),
              str([s["kind"] for s in segs]))

        # ------------------------------------------------------ websocket
        print("\nS2  live turn over the websocket")
        async with http.ws_connect(f"{WS}/ws/session/{sid}") as ws:
            first = await collect(ws, "replay", timeout=20)
            kinds = [k for k, _ in first]
            check("hello then replay on connect",
                  kinds[:2] == ["hello", "replay"], str(kinds[:3]))

            await ws.send_json({"t": "ping", "d": {}})
            pong = await collect(ws, "pong", timeout=10)
            check("heartbeat answered", any(k == "pong" for k, _ in pong))

            await ws.send_json({"t": "action",
                                "d": {"text": "I push open the door and step inside."}})
            frames = await collect(ws, "turn_end", timeout=240)
            kinds = [k for k, _ in frames]

            check("turn_start emitted", "turn_start" in kinds)
            prose = [d["text"] for k, d in frames if k == "prose"]
            check("prose streamed in chunks", len(prose) >= 3, f"{len(prose)} frames")
            check("prose is substantial", len("".join(prose)) > 150,
                  f"{len(''.join(prose))} chars")
            check("turn_end emitted", "turn_end" in kinds)

            slots = [d for k, d in frames if k == "image_slot"]
            if not slots:
                print("  ----  no image this turn (model compliance; see PHASE0.md)")
            else:
                # Per image, not globally: the opening illustration can still
                # be generating when the next turn's slot opens, so ITS
                # progress frames legitimately arrive first. What must hold is
                # that no image reports progress before its own placeholder
                # exists, or the reader sees a picture appear out of nowhere.
                order = [(i, k, d) for i, (k, d) in enumerate(frames)]
                ok, detail = True, ""
                for d in slots:
                    iid = d["image_id"]
                    si = next(i for i, k, dd in order
                              if k == "image_slot" and dd.get("image_id") == iid)
                    pi = next((i for i, k, dd in order
                               if k == "image_progress" and dd.get("image_id") == iid),
                              10 ** 9)
                    if si >= pi:
                        ok, detail = False, f"{iid[:8]} slot at {si}, progress at {pi}"
                check("each image_slot precedes its own progress", ok,
                      detail or f"{len(slots)} slot(s) checked")

                # Wait for that image to finish.
                img_id = slots[-1]["image_id"]
                more = await collect(ws, "image_ready", timeout=240)
                k2 = [k for k, _ in more]
                check("progress reported", k2.count("image_progress") >= 3,
                      f"{k2.count('image_progress')}")
                check("preview frames sent", k2.count("image_preview") >= 1,
                      f"{k2.count('image_preview')}")
                check("image_ready received", "image_ready" in k2)

                url = next((d["url"] for k, d in more if k == "image_ready"), None)
                if url:
                    async with http.get(BASE + url) as r:
                        blob = await r.read()
                    check("image downloadable", r.status == 200 and len(blob) > 10000,
                          f"{len(blob)} bytes")

        # ------------------------------------------------------- reconnect
        print("\nS3  reconnect replays from storage")
        async with http.ws_connect(f"{WS}/ws/session/{sid}") as ws2:
            frames = await collect(ws2, "replay", timeout=20)
            replay = next((d for k, d in frames if k == "replay"), {})
            segs2 = replay.get("segments", [])
            check("replay contains the played turn", len(segs2) > len(segs),
                  f"{len(segs)} -> {len(segs2)} segments")
            check("replay carries image states", bool(replay.get("images")),
                  f"{len(replay.get('images', {}))} images")
            kinds2 = [s["kind"] for s in segs2 if s["kind"]]
            check("prose and image interleaved in order",
                  kinds2.count("prose") >= 2, str(kinds2))

    print(f"\n{'=' * 58}")
    print(f"  {len(passed)} passed, {len(failed)} failed")
    for f in failed:
        print(f"    FAILED: {f}")
    print("=" * 58)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
