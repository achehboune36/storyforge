"""Image generation jobs: queue, run, report, preempt, resume.

Borrows gameshow's cancellable-task registry (`game_engine.py:80-105`),
including the detail that makes it correct: **re-validate after every await**.
A task that resumes into a world that has moved on must discard its result
rather than broadcast it. Here that means an image finishing after its session
was abandoned, or after the slot it belongs to was superseded.

It also fixes game-poc's structural mistake. There,
`_generate_and_send_image(websocket, ...)` closed over the browser socket, so
an image that finished after a disconnect was simply lost. Jobs here are owned
by the session and write to the database; delivery is a separate concern, so a
reconnecting client replays whatever completed while it was away.

Retry deliberately perturbs the seed. ComfyUI serves an identical workflow from
cache instantly, so a naive "regenerate" hands back the very same picture — a
hazard that produced three misleading results during Phase 0 before it was
understood.
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from backend import database as db
from backend import image_prompt as ip
from backend import models
from backend import prompt_book as book
from backend import ratings
from backend import workflow as wf
from backend.comfy_client import ComfyClient
from backend.gpu_broker import Consumer, GpuBroker
from backend.turn_parser import ImageSlot

# The model emits ids loosely: "ohanji (the woman)" was registered verbatim as
# a place with no appearance, so that character could never be visually
# consistent. Ids are normalised, and a parenthetical is treated as a hint that
# this is a person, not a location.
_ID_RE = re.compile(r"[^a-z0-9_]+")


def clean_entity_key(raw: str) -> tuple[str, str]:
    """-> (key, kind_hint). kind_hint is '' when nothing suggests a person."""
    text = (raw or "").strip().lower()
    hint = ""
    paren = re.search(r"\(([^)]*)\)", text)
    if paren:
        inner = paren.group(1)
        if any(w in inner for w in ("woman", "man", "girl", "boy", "person",
                                    "stranger", "child", "she", "he")):
            hint = "character"
        text = text[:paren.start()].strip()
    key = _ID_RE.sub("_", text).strip("_")[:40]
    return key, hint

# Underscores are part of an id, so _ID_RE keeps them — splitting an id into
# words needs its own pattern or "headmasters_office" stays one token.
_WORD_RE = re.compile(r"[^a-z0-9]+")
_STOPWORDS = {"the", "a", "an", "of", "and", "one", "to", "in", "at", "for", ""}
# Words shared by several roles. "magic_head" must resolve to elena on the
# strength of "magic"; counting "head" as well made it ambiguous against all
# three Head Instructors, so the model's invented id became a new entity.
_ROLE_NOISE = {"head", "instructor", "department", "teacher", "professor",
               "master", "student", "year", "1st", "2nd", "3rd", "academy"}


def _words(text: str) -> set[str]:
    return set(_WORD_RE.split((text or "").lower()))


def resolve_alias(key: str, known: dict[str, dict]) -> str:
    """Map an invented id onto a declared entity, or return it unchanged.

    The narrator is given the roster as `emily (Headmaster, …)` and still
    writes `headmaster`. Taken literally that creates a brand new entity with
    no appearance, so the real character's tags never reach the prompt and the
    model invents someone — it drew the headmaster as a man. A key that
    matches exactly one declared entity's role or id is that entity.
    """
    if key in known:
        return key
    words = _words(key) - _STOPWORDS
    if not words:
        return key
    strong = words - _ROLE_NOISE
    for probe in (strong, words):      # distinctive words first, then anything
        if not probe:
            continue
        hits = [k for k, e in known.items()
                if probe & (_words(f"{k} {e.get('role') or ''}") - _STOPWORDS)]
        if len(hits) == 1:
            return hits[0]
        if hits:
            break
    else:
        hits = []
    if not hits:
        # Shortened names: the narrator writes "mira" for mirabel. Only on a
        # unique prefix, and only from four characters, so "la" can't claim
        # laris.
        short = [k for k in known if len(key) >= 4 and k.startswith(key)]
        if len(short) == 1:
            return short[0]
    # Ambiguity is worse than a new entity: two head instructors both match
    # "head", and picking either silently puts the wrong face in the frame.
    return key


EventCb = Callable[[str, dict], Awaitable[None]]

IMAGES_DIR = Path(__file__).resolve().parent.parent / "data" / "images"

# Preview frames arrive one per sampler step and bypass ComfyUI's own
# throttling, so they are throttled here before touching a socket.
PREVIEW_MIN_INTERVAL_S = 1.5

PORTRAITS_DIR = Path(__file__).resolve().parent.parent / "data" / "portraits"

# Queue priority for the refine pass. Anything a reader is waiting on is 0.
HIRES_PRIORITY = 10


def model_name(img_cfg: dict) -> str:
    """The file that defines the look, whichever architecture is configured.

    SDXL names one `checkpoint`; FLUX/Anima name a separate `unet`. Recorded on
    every image row so an old picture can always be traced to what drew it.
    """
    return img_cfg.get("checkpoint") or img_cfg.get("unet") or "unknown"

# The reference portrait's wording lives in prompting.toml under [portrait]:
# a plain head-and-shoulders shot, so PuLID has a clean face to lock onto.


@dataclass
class _Live:
    image_id: str
    session_id: str
    task: asyncio.Task
    kind: str = "image"      # "image" (a reader is waiting) | "hires"
    started: float = field(default_factory=time.time)
    step: int = 0
    total: int = 0
    s_per_step: float = 1.0
    comfy_prompt_id: str | None = None

    def remaining_s(self) -> float:
        if self.total <= 0:
            return 999.0
        return max(0.0, (self.total - self.step) * self.s_per_step)


class ImageJobRunner:
    def __init__(self, cfg: dict, broker: GpuBroker, comfy: ComfyClient,
                 on_event: EventCb | None = None):
        self.cfg = cfg
        self.broker = broker
        self.comfy = comfy
        self.on_event = on_event
        self._live: _Live | None = None
        # Priority queue, because the refine pass must never make a reader
        # wait: a story image is 0, a second pass is 10.
        self._queue: asyncio.PriorityQueue[tuple[int, int, str, str]] = \
            asyncio.PriorityQueue()
        self._seq = 0
        self._worker: asyncio.Task | None = None

        broker.image_progress_hint = self._progress_hint
        broker.image_cancel = self._cancel_live

    def _enqueue(self, image_id: str, *, prio: int = 0, kind: str = "image") -> None:
        """Queue a pass. `_seq` keeps equal priorities in arrival order and
        stops the tuple comparison ever reaching the ids."""
        self._seq += 1
        self._queue.put_nowait((prio, self._seq, kind, image_id))

    @property
    def img_cfg(self) -> dict:
        """`config.image` merged with the ACTIVE checkpoint's own settings.

        Read fresh on every use, never cached: switching the model mid-story
        must reach the next image, and a model that needs v-prediction or a
        different CFG has to bring those with it.
        """
        return models.image_cfg(self.cfg)

    # ------------------------------------------------------------- lifecycle

    def start(self) -> None:
        if self._worker is None or self._worker.done():
            self._worker = asyncio.create_task(self._run_worker())

    async def stop(self) -> None:
        if self._worker:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
        await self._cancel_live()

    async def resume_pending(self) -> int:
        """Re-queue anything left over from a previous process."""
        n = db.reconcile_startup()
        for job in db.pending_jobs():
            self._enqueue(job["image_id"])
        return n

    # --------------------------------------------------------------- submit

    # ----------------------------------------------------------- portraits

    def negative_for(self, session_id: str, checkpoint: str = "") -> str:
        """This session's negative prompt, resolved NOW.

        Sessions store the negative they were created with, which means a
        story already in play would never see an edit to prompting.toml. The
        rating and the story's own `negative` field are stable; the wording
        around them is not, so it is rebuilt per image. What was actually used
        is still recorded on the image row.
        """
        sess = db.get_session(session_id) or {}
        story = db.get_story(sess.get("story_id") or "") or {}
        return ratings.negative_prompt(sess.get("rating"),
                                       story.get("negative", ""),
                                       checkpoint or model_name(self.img_cfg))

    async def ensure_portrait(self, session_id: str, key: str, *, style: str,
                              negative: str | None = None) -> str | None:
        """Generate this character's reference portrait if they lack one.

        Called in the background the first time a character appears. Returns
        the ComfyUI-visible filename, or None if the entity has no appearance
        to draw (invented ids start empty) or generation failed.
        """
        ent = db.get_entity(session_id, key)
        if not ent or ent.get("kind") != "character":
            return None
        if ent.get("ref_image_id"):
            return ent["ref_image_id"]
        appearance = (ent.get("appearance") or "").strip()
        if not appearance:
            return None

        icfg = self.img_cfg
        pcfg = icfg.get("portrait") or {}
        if not pcfg.get("enabled"):
            return None

        prompt = book.portrait_template().format(appearance=appearance,
                                                 style=style)
        graph = wf.build_sdxl({**icfg, "pulid": {"enabled": False}})
        vals = wf.values({**icfg,
                          "width": pcfg.get("width", icfg["width"]),
                          "height": pcfg.get("height", icfg["height"]),
                          "steps": pcfg.get("steps", icfg["steps"])},
                         prompt=prompt,
                         negative=(negative if negative is not None
                                   else self.negative_for(session_id)),
                         seed=ip.entity_seed(session_id, key),
                         prefix=f"portrait_{key}")
        res = await self.comfy.generate(wf.patch(graph, vals), previews=False)
        if not res.ok or not res.images:
            return None

        name = res.images[0]["filename"]
        # PuLID's LoadImage reads from ComfyUI's input dir, so the portrait has
        # to live there, not only in our own storage.
        dest = PORTRAITS_DIR / session_id / name
        await self.comfy.download(res.images[0], dest)
        await self.comfy.stage_input(dest, name)
        db.set_entity_ref(session_id, key, name)
        await self._emit("portrait_ready", {"session_id": session_id, "key": key,
                                            "file": name})
        return name

    async def submit(self, *, session_id: str, turn_id: str, turn_idx: int,
                     slot: ImageSlot, style: str,
                     negative: str | None = None) -> str:
        """Create the image row and enqueue it. Returns the image id.

        Called the moment a directive closes in the stream — before generation
        starts — so the UI can open a placeholder in the right position.
        """
        known = {e["key"]: e for e in db.get_entities(session_id)}

        # The model invents entity ids (it produced rooftop/morgue/docks
        # unprompted in testing). Register them so they get a stable seed from
        # now on; appearance stays empty until something fills it.
        cleaned: list[str] = []
        for raw_key in slot.entities:
            key, hint = clean_entity_key(raw_key)
            if not key:
                continue
            key = resolve_alias(key, known)
            if key in cleaned:
                continue
            cleaned.append(key)
            if key not in known:
                db.upsert_entity(session_id, key, kind=hint or "place", role="",
                                 appearance="", seed=ip.entity_seed(session_id, key))
        slot = ImageSlot(index=slot.index, entities=cleaned,
                         keywords=slot.keywords, framing=slot.framing)
        known = {e["key"]: e for e in db.get_entities(session_id)}

        icfg = self.img_cfg
        sess = db.get_session(session_id) or {}
        plan = ip.plan_image(session_id=session_id, turn_idx=turn_idx,
                             slot=slot.index, keywords=slot.keywords,
                             entity_keys=slot.entities, known=known, style=style,
                             rating=(sess["rating"] if "rating" in sess.keys()
                                     else "13+"),
                             unet_name=model_name(icfg),
                             arch=icfg.get("arch", ""),
                             framing=slot.framing,
                             width=icfg["width"], height=icfg["height"])

        image_id = db.create_image(
            session_id, turn_id, slot_pos=slot.index, keywords=slot.keywords,
            entities=plan["entities"], prompt=plan["prompt"], seed=plan["seed"],
            negative=(negative if negative is not None
                      else self.negative_for(session_id)),
            width=plan["width"], height=plan["height"],
            steps=icfg["steps"], model=model_name(icfg))
        db.enqueue_job(session_id, image_id)
        self._enqueue(image_id)
        # A refine can hold the GPU for minutes. The queue priority alone does
        # not help once one is already running, so bump it.
        if self._live is not None and self._live.kind == "hires":
            await self._cancel_live()

        await self._emit("image_state", {
            "image_id": image_id, "status": "queued",
            "queue_len": self._queue.qsize(),
            # The placeholder must open at the size this image will be, or the
            # reading column jumps when a landscape frame lands in a portrait
            # hole.
            "width": plan["width"], "height": plan["height"],
        })
        return image_id

    async def retry(self, image_id: str) -> bool:
        img = db.get_image(image_id)
        if not img or img["status"] in ("running",):
            return False
        # Perturb the seed: an identical workflow is served from cache and
        # would return the exact image the user just rejected.
        db.update_image(image_id, status="queued", error=None,
                        seed=(int(img["seed"]) + 1) % (2 ** 31 - 1))
        db.enqueue_job(img["session_id"], image_id, priority=3)
        self._enqueue(image_id)
        await self._emit("image_state", {"image_id": image_id, "status": "queued"})
        return True

    # --------------------------------------------------------------- worker

    async def _run_hires(self, image_id: str) -> None:
        """Re-sample a finished image larger, in place, after the fact.

        Everything here is deliberately give-way: it runs at the back of the
        queue, it is interrupted the moment a reader needs the GPU, and every
        failure path leaves the original image exactly as it was.
        """
        img = db.get_image(image_id)
        if not img or img["status"] != "ready" or not img["file_path"]:
            return
        if (img["hires"] or "") == "done":
            return
        src = Path(img["file_path"])
        if not src.exists():
            db.update_image(image_id, hires="failed")
            return

        # The model RECORDED ON THE ROW, not the active one: refining with a
        # model the reader switched to since would hand them a different
        # picture, not a sharper one.
        icfg = models.image_cfg(self.cfg, img["model"] or None)
        scale = float(icfg.get("hires_scale") or 0)
        if scale <= 1:
            db.update_image(image_id, hires="skip")
            return

        # `hires_model` lets the second pass run a DIFFERENT model, which is
        # the whole point of the hybrid: let the model that gets four people
        # in the right clothes compose the scene, then let the model with the
        # style the reader chose repaint it. At this denoise the composition
        # survives and the look comes from the second model's weights.
        prompt, negative = img["prompt"], img["negative"] or ""
        rname = icfg.get("hires_model") or img["model"] or None
        rcfg = icfg
        if rname and rname != img["model"]:
            rcfg = models.image_cfg(self.cfg, rname)
            # The policy belongs to the model that asked for the pass.
            for k in ("hires_scale", "hires_steps", "hires_denoise", "hires_method"):
                if icfg.get(k) is not None:
                    rcfg[k] = icfg[k]
            # Both prompt and negative were written for the first model's
            # vocabulary; the repaint needs the second one's.
            q = book.quality(book.family(rcfg.get("arch", ""), rname), rname)
            if q and q not in prompt:
                prompt = f"{q}, {prompt}"
            negative = self.negative_for(img["session_id"], checkpoint=rname)

        staged = f"sfhr_{image_id}.png"
        await self.comfy.stage_input(src, staged)
        graph = wf.build_hires(rcfg)
        vals = wf.hires_values(rcfg, image=staged, prompt=prompt,
                               negative=negative,
                               # +1: an identical workflow is served from
                               # ComfyUI's cache, which would return the very
                               # image we are trying to improve.
                               seed=int(img["seed"]) + 1,
                               prefix=f"sfhr_{img['session_id'][:8]}")
        patched = wf.patch(graph, vals)
        steps = int(rcfg.get("hires_steps") or rcfg["steps"]) or 1

        async with self.broker.lease(Consumer.IMAGE, img["session_id"]):
            db.update_image(image_id, hires="running")
            self._live = _Live(image_id=image_id, session_id=img["session_id"],
                               task=asyncio.current_task(), total=steps,
                               kind="hires", s_per_step=1.0)
            try:
                res = await self.comfy.generate(patched)
            finally:
                self._live = None

        if res.interrupted:
            # A reader needed the GPU. Go to the back of the queue again.
            db.update_image(image_id, hires="queued")
            self._enqueue(image_id, prio=HIRES_PRIORITY, kind="hires")
            return
        if not res.ok or not res.images:
            db.update_image(image_id, hires="failed")
            print(f"  hires pass failed for {image_id[:8]}: {res.error or 'no image'}")
            return

        current = db.get_image(image_id)
        if not current or current["status"] != "ready":
            return          # deleted or retried underneath us

        dest = IMAGES_DIR / img["session_id"] / f"{image_id}_hr.png"
        await self.comfy.download(res.images[0], dest)
        # The original file is kept: it is what a failed or reverted refine
        # falls back to, and it costs a few hundred KB.
        db.update_image(image_id, hires="done", file_path=str(dest))
        w = h = 0
        try:
            from PIL import Image
            with Image.open(dest) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001 — the size is a nicety, not the point
            pass
        await self._emit("image_hires_ready", {
            "image_id": image_id,
            "url": f"/media/{img['session_id']}/{image_id}_hr.png",
            "width": w or int(img["width"]), "height": h or int(img["height"]),
            "wall_s": round(res.wall_s, 1),
        })

    async def _run_worker(self) -> None:
        while True:
            _prio, _seq, kind, image_id = await self._queue.get()
            try:
                if kind == "hires":
                    await self._run_hires(image_id)
                else:
                    await self._run_one(image_id)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                if kind == "hires":
                    # The reader already has a good picture; a failed refine
                    # is a logged disappointment, not a broken image.
                    db.update_image(image_id, hires="failed")
                    print(f"  hires pass failed for {image_id[:8]}: {e}")
                else:
                    db.update_image(image_id, status="failed", error=str(e)[:400],
                                    finished_at=db.now())
                    await self._emit("image_failed", {
                        "image_id": image_id, "error": str(e)[:200],
                        "retryable": True})
            finally:
                self._queue.task_done()

    async def _run_one(self, image_id: str) -> None:
        img = db.get_image(image_id)
        if not img or img["status"] in ("cancelled", "ready"):
            return   # superseded while queued

        # If a character in this shot has a reference portrait, lock the face
        # to it. Otherwise fall back to seed + frozen appearance alone.
        reference = None
        for key in img["entities"]:
            ent = db.get_entity(img["session_id"], key)
            if ent and ent.get("ref_image_id"):
                reference = ent["ref_image_id"]
                break
        icfg = self.img_cfg
        graph = (wf.build_sdxl(icfg, reference=reference)
                 if reference and icfg.get("arch") == "sdxl"
                 else wf.build(icfg))
        vals = wf.values(icfg, prompt=img["prompt"],
                         negative=img["negative"] or "",
                         seed=int(img["seed"]),
                         # Per image, not per model: the narrator asked for
                         # this shape and the row recorded it.
                         width=int(img["width"]) or None,
                         height=int(img["height"]) or None,
                         prefix=f"sf_{img['session_id'][:8]}")
        patched = wf.patch(graph, vals)

        last_preview = 0.0
        expected = float(icfg.get("expected_wall_s", 14))
        steps = int(icfg["steps"]) or 1

        async def on_state(s: str) -> None:
            await self._emit("image_state", {"image_id": image_id, "status": s})

        async def on_progress(v: int, m: int) -> None:
            if self._live:
                self._live.step, self._live.total = v, m
                elapsed = time.time() - self._live.started
                if v > 0:
                    self._live.s_per_step = elapsed / v
            await self._emit("image_progress", {
                "image_id": image_id, "value": v, "max": m,
                "pct": int(100 * v / m) if m else 0})

        async def on_preview(jpeg: bytes) -> None:
            nonlocal last_preview
            now = time.time()
            if now - last_preview < PREVIEW_MIN_INTERVAL_S:
                return
            last_preview = now
            import base64
            await self._emit("image_preview", {
                "image_id": image_id,
                "data_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode(),
            })

        async with self.broker.lease(Consumer.IMAGE, img["session_id"]):
            db.update_image(image_id, status="running", started_at=db.now(),
                            attempts=int(img["attempts"]) + 1)
            task = asyncio.current_task()
            self._live = _Live(image_id=image_id, session_id=img["session_id"],
                               task=task, total=steps,
                               s_per_step=expected / steps)
            try:
                res = await self.comfy.generate(
                    patched, on_progress=on_progress, on_preview=on_preview,
                    on_state=on_state)
            finally:
                self._live = None

        # Re-validate after the await: the world may have moved on.
        current = db.get_image(image_id)
        if not current or current["status"] == "cancelled":
            return

        if res.interrupted:
            # Preempted for a user turn. Seed and prompt are persisted, and
            # Phase 1 proved a re-run is pixel-identical, so nothing is lost.
            db.update_image(image_id, status="queued", error=None)
            db.enqueue_job(img["session_id"], image_id, priority=4)
            self._enqueue(image_id)
            await self._emit("image_state", {"image_id": image_id,
                                             "status": "queued",
                                             "reason": "preempted"})
            return

        if not res.ok or not res.images:
            db.update_image(image_id, status="failed",
                            error=res.error or "no image returned",
                            finished_at=db.now())
            await self._emit("image_failed", {
                "image_id": image_id, "error": (res.error or "")[:200],
                "retryable": True})
            return

        dest = IMAGES_DIR / img["session_id"] / f"{image_id}.png"
        await self.comfy.download(res.images[0], dest)
        db.update_image(image_id, status="ready", file_path=str(dest),
                        comfy_prompt_id=res.prompt_id, finished_at=db.now())
        await self._emit("image_ready", {
            "image_id": image_id,
            "url": f"/media/{img['session_id']}/{image_id}.png",
            "width": img["width"], "height": img["height"],
            "wall_s": round(res.wall_s, 1),
        })

        # The reader now has a picture. If this model asks for a second pass,
        # queue it behind everything else: it replaces the file underneath
        # them later, and nobody waits for it.
        if float(icfg.get("hires_scale") or 0) > 1:
            db.update_image(image_id, hires="queued")
            self._enqueue(image_id, prio=HIRES_PRIORITY, kind="hires")

    # ------------------------------------------------------ broker callbacks

    def _progress_hint(self) -> tuple[float, str] | None:
        if not self._live:
            return None
        return self._live.remaining_s(), self._live.image_id

    async def _cancel_live(self) -> None:
        """Interrupt the running image so a user turn can have the GPU."""
        live = self._live
        if not live:
            return
        with contextlib.suppress(Exception):
            await self.comfy.interrupt(live.comfy_prompt_id)

    async def cancel_session(self, session_id: str) -> None:
        live = self._live
        if live and live.session_id == session_id:
            await self._cancel_live()

    # ---------------------------------------------------------------- events

    async def _emit(self, kind: str, payload: dict) -> None:
        if self.on_event:
            await self.on_event(kind, payload)
