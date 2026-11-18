"""Per-session runtime: owns the turn, fans out to whichever sockets are attached.

The structural rule here is the fix for game-poc's central mistake. There,
`_generate_and_send_image(websocket, ...)` closed over the browser socket, so
work that finished after a disconnect was simply lost. Here **the session owns
the turn task**; sockets attach to and detach from it. Closing a laptop lid
mid-turn loses nothing — the prose keeps streaming into the database and the
image keeps generating, and a reconnecting client replays from storage.

Two behaviours worth knowing about:

* **Repair is gated on cadence, not on the turn.** Directive compliance from
  llama3.2:3b is probabilistic (~50-75%), so a turn with no directive is
  normal. But a story going several turns with no picture is not. Asking for
  keywords after *every* directive-less turn was measured to over-trigger
  badly — on pure dialogue it invented junk like "glass, clock, Sal, hands"
  two times in three. Waiting for a run of quiet turns avoids that while still
  keeping the story illustrated.

* **Summarisation runs only when the GPU is idle**, and never in front of a
  user turn.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass, field
from typing import Any

from backend import context, database as db
from backend.gpu_broker import Consumer, GpuBroker
from backend.image_jobs import ImageJobRunner
from backend.llm_ollama import (OllamaClient, build_messages, request_keywords,
                                stream_turn)
from backend.turn_parser import ImageSlot, ProseChunk, segments_from_events

# Turns without a picture before the repair call is considered.
REPAIR_AFTER_QUIET_TURNS = 3


@dataclass
class Socket:
    """One attached client. Its own queue so a slow reader cannot stall a turn."""
    ws: Any
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    writer: asyncio.Task | None = None


class SessionRuntime:
    def __init__(self, session_id: str, cfg: dict, *, broker: GpuBroker,
                 llm: OllamaClient, jobs: ImageJobRunner):
        self.id = session_id
        self.cfg = cfg
        self.broker = broker
        self.llm = llm
        self.jobs = jobs
        self.sockets: list[Socket] = []
        self._turn_task: asyncio.Task | None = None
        self._seq = 0
        self._quiet_turns = 0
        self._last_keywords: str | None = None
        self.busy = False

    # ------------------------------------------------------------- transport

    async def attach(self, ws) -> Socket:
        sock = Socket(ws=ws)
        sock.writer = asyncio.create_task(self._writer(sock))
        self.sockets.append(sock)
        await self._send_one(sock, "hello", {
            "session": db.get_session(self.id),
            "busy": self.busy,
        })
        await self._send_one(sock, "replay", self.replay_payload())
        return sock

    async def detach(self, sock: Socket) -> None:
        if sock in self.sockets:
            self.sockets.remove(sock)
        if sock.writer:
            sock.writer.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sock.writer

    # Coalescing exists for backpressure, not as a default. Merging whatever
    # happens to be queued turned 84 streamed deltas into two or three blobs,
    # so the reader saw the turn appear all at once instead of being written.
    # Only merge once the queue is genuinely deep.
    COALESCE_ABOVE = 8
    COALESCE_MAX = 24

    async def _writer(self, sock: Socket) -> None:
        """One writer per socket. Streams frames through; merges only under load."""
        while True:
            t, d = await sock.queue.get()
            batch = [(t, d)]
            if sock.queue.qsize() > self.COALESCE_ABOVE:
                while not sock.queue.empty() and len(batch) < self.COALESCE_MAX:
                    batch.append(sock.queue.get_nowait())
            # Merge runs of prose in the same turn into one frame.
            out: list[tuple[str, dict]] = []
            for kind, data in batch:
                if (kind == "prose" and out and out[-1][0] == "prose"
                        and out[-1][1].get("turn_id") == data.get("turn_id")):
                    out[-1][1]["text"] += data["text"]
                else:
                    out.append((kind, dict(data)))
            for kind, data in out:
                with contextlib.suppress(Exception):
                    await sock.ws.send_json({"t": kind, "d": data})

    async def _emit(self, kind: str, data: dict) -> None:
        self._seq += 1
        data = {**data, "seq": self._seq}
        # Snapshot: gameshow's broadcast iterated the live list while awaiting,
        # which raises if a client connects or disconnects mid-loop.
        for sock in list(self.sockets):
            try:
                sock.queue.put_nowait((kind, data))
            except asyncio.QueueFull:
                # Lossy by nature; never drop prose or a finished image.
                if kind in ("image_progress", "image_preview"):
                    continue
                with contextlib.suppress(Exception):
                    _ = sock.queue.get_nowait()
                    sock.queue.put_nowait((kind, data))

    async def _send_one(self, sock: Socket, kind: str, data: dict) -> None:
        with contextlib.suppress(Exception):
            await sock.ws.send_json({"t": kind, "d": data})

    # ---------------------------------------------------------------- replay

    def replay_payload(self) -> dict:
        """Everything needed to rebuild the page from storage."""
        segs = db.get_segments(self.id)
        images = {i["id"]: {
            "image_id": i["id"], "status": i["status"],
            "url": (f"/media/{self.id}/{db.image_file_name(i)}"
                    if i["status"] == "ready" else None),
            "width": i["width"], "height": i["height"],
            "error": i["error"],
        } for i in db.get_images(self.id)}
        return {"segments": segs, "images": images, "busy": self.busy}

    # ----------------------------------------------------------------- turns

    async def submit_action(self, text: str, *, passing: bool = False) -> bool:
        if self.busy:
            await self._emit("error", {"code": "busy",
                                       "message": "a turn is already running"})
            return False
        self._turn_task = asyncio.create_task(self._run_turn(text, passing=passing))
        return True

    async def cancel_turn(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task

    async def _run_turn(self, action: str, *, passing: bool = False) -> None:
        self.busy = True
        sess = db.get_session(self.id)
        try:
            db.touch_session(self.id)
            # A pass writes no player turn — the reader did not say anything,
            # and replaying "(you let the moment pass)" as if it were dialogue
            # would put words in their mouth on every reload.
            if not passing:
                user_turn = db.add_turn(self.id, "user", action)
            # The user's own words are part of the story and must replay with
            # it. Without a segment row they render as a NULL-kind join result
            # and the client skips them — the turn silently disappears.
                db.add_segments(user_turn, [{"kind": "prose", "text": action}])
            await self._emit("turn_start", {"action": action, "passing": passing})

            asm = context.assemble(self.id, action,
                                   num_ctx=self.cfg["llm"]["num_ctx"],
                                   passing=passing)
            turn_idx = db.next_turn_idx(self.id)
            turn_id = db.add_turn(self.id, "narrator", "", idx=turn_idx)

            slots: list[ImageSlot] = []
            slot_image_ids: list[str] = []
            pieces: list[str] = []

            async def on_event(ev) -> None:
                if isinstance(ev, ProseChunk):
                    pieces.append(ev.text)
                    await self._emit("prose", {"turn_id": turn_id, "text": ev.text})
                else:
                    slots.append(ev)
                    # The placeholder must open at this exact position, before
                    # generation starts — that is what makes reading unblocked.
                    image_id = await self.jobs.submit(
                        session_id=self.id, turn_id=turn_id, turn_idx=turn_idx,
                        slot=ev, style=sess["style"])
                    slot_image_ids.append(image_id)
                    # Size comes from the IMAGE ROW, never from config: the
                    # narrator picks the frame shape per picture now, and a
                    # placeholder opened at the wrong shape is not just ugly
                    # — the figure crops the image to fit it.
                    row = db.get_image(image_id) or {}
                    ev_payload = {"turn_id": turn_id, "pos": ev.index,
                                  "image_id": image_id,
                                  "width": row.get("width") or self.cfg["image"]["width"],
                                  "height": row.get("height") or self.cfg["image"]["height"]}
                    await self._emit("image_slot", ev_payload)
                    self._last_keywords = ev.keywords

            async with self.broker.lease(Consumer.LLM, self.id):
                res = await stream_turn(
                    self.llm, build_messages(asm.system, asm.history, asm.user),
                    on_event=on_event,
                    max_images=self.cfg["image"]["max_images_per_turn"],
                    previous_keywords=[i["keywords"] for i in
                                       db.get_images(self.id) if i["keywords"]])

                if not res.ok:
                    await self._emit("error", {"code": "llm",
                                               "message": res.llm.error or "llm failed"})
                    return

                if res.llm.truncated:
                    # The reply hit the output cap and stops mid-sentence.
                    # It used to be saved exactly like a finished turn, so a
                    # story just ended in the middle of a word with nothing
                    # said about it.
                    print(f"[session {self.id[:8]}] reply TRUNCATED at the output "
                          f"cap ({res.llm.eval_count} tokens) — raise "
                          f"llm.num_predict", flush=True)
                    await self._emit("warning", {
                        "code": "truncated",
                        "message": "the narrator hit its output limit and stopped "
                                   "mid-sentence — undo this turn and retry"})

                prose = "".join(pieces)
                # Repair, gated on cadence — see module docstring.
                if not slots:
                    self._quiet_turns += 1
                    if self._quiet_turns >= REPAIR_AFTER_QUIET_TURNS:
                        kw = await request_keywords(self.llm, prose[-800:])
                        if kw:
                            self._quiet_turns = 0
                            slot = ImageSlot(index=0, entities=[], keywords=kw)
                            image_id = await self.jobs.submit(
                                session_id=self.id, turn_id=turn_id,
                                turn_idx=turn_idx, slot=slot,
                                style=sess["style"])
                            slots.append(slot)
                            slot_image_ids.append(image_id)
                            row = db.get_image(image_id) or {}
                            await self._emit("image_slot", {
                                "turn_id": turn_id, "pos": 0, "image_id": image_id,
                                "width": row.get("width") or self.cfg["image"]["width"],
                                "height": row.get("height") or self.cfg["image"]["height"],
                                "repaired": True})
                            self._last_keywords = kw
                else:
                    self._quiet_turns = 0

            # Persist the finished turn.
            # Was: looked the ids back up out of db.get_images() by the wrong
            # key, raising KeyError on EVERY turn that produced an image. The
            # broad except below swallowed it, so those turns were written with
            # empty text and no segments — silent, total loss of the turn.
            # Three turns of a real playthrough were destroyed this way.
            # The ids are known at submit time; keep them.
            segs = segments_from_events(res.events)
            rows = []
            k = 0
            for seg in segs:
                if seg["kind"] == "image":
                    rows.append({"kind": "image",
                                 "image_id": slot_image_ids[k]
                                 if k < len(slot_image_ids) else None})
                    k += 1
                else:
                    rows.append({"kind": "prose", "text": seg["text"]})
            db.connect().execute("UPDATE turns SET text=?, raw_text=? WHERE id=?",
                                 (prose, res.llm.text, turn_id))
            db.connect().commit()
            db.add_segments(turn_id, rows)

            await self._emit("turn_end", {"turn_id": turn_id,
                                          "images": len(slots)})
        except asyncio.CancelledError:
            await self._emit("error", {"code": "cancelled", "message": "turn cancelled"})
            raise
        except Exception as e:  # noqa: BLE001
            # Loud: a swallowed exception here cost three turns of a real
            # playthrough before anyone noticed.
            import traceback
            print(f"[session {self.id[:8]}] TURN FAILED: {e!r}", flush=True)
            traceback.print_exc()
            await self._emit("error", {"code": "internal", "message": str(e)[:200]})
        finally:
            self.busy = False
            asyncio.create_task(self._maybe_summarise())

    # ------------------------------------------------------------- summarise

    async def _maybe_summarise(self) -> None:
        """Background, lowest priority, only while the GPU is quiet."""
        await asyncio.sleep(2)
        if self.busy or not context.summary_due(self.id):
            return
        try:
            msgs, through = context.build_summary_messages(self.id)
            async with self.broker.lease(Consumer.LLM, self.id):
                res = await self.llm.stream(msgs, num_predict=260, temperature=0.3)
                if res.ok and res.text.strip():
                    db.set_summary(self.id, res.text.strip(), through)

                turns = db.get_turns(self.id)
                passage = "\n".join(
                    context.strip_directives(t["text"]) for t in turns[-6:])
                facts = await self.llm.stream(
                    context.build_facts_messages(passage),
                    num_predict=120, temperature=0.2)
                if facts.ok:
                    for f in context.parse_facts(facts.text):
                        db.add_memory(self.id, f, importance=7)
            await self._emit("status", {"summarised_through": through})
        except Exception:
            pass   # never let housekeeping break a session


class SessionManager:
    def __init__(self, cfg: dict, broker: GpuBroker, llm: OllamaClient,
                 jobs: ImageJobRunner):
        self.cfg, self.broker, self.llm, self.jobs = cfg, broker, llm, jobs
        self._runtimes: dict[str, SessionRuntime] = {}

    def get(self, session_id: str) -> SessionRuntime:
        rt = self._runtimes.get(session_id)
        if rt is None:
            rt = SessionRuntime(session_id, self.cfg, broker=self.broker,
                                llm=self.llm, jobs=self.jobs)
            self._runtimes[session_id] = rt
        return rt

    def find_by_image(self, image_id: str) -> SessionRuntime | None:
        img = db.get_image(image_id)
        return self.get(img["session_id"]) if img else None

    async def broadcast_image_event(self, kind: str, payload: dict) -> None:
        rt = self.find_by_image(payload.get("image_id", ""))
        if rt:
            await rt._emit(kind, payload)

    async def broadcast_all(self, kind: str, payload: dict) -> None:
        """Tell every attached reader something that is not about one image —
        today, that the image model was switched out from under them."""
        for rt in list(self._runtimes.values()):
            await rt._emit(kind, payload)
