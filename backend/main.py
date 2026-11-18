"""StoryForge server.

    python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8100

REST creates and lists things; the WebSocket carries everything live. That
split is borrowed from gameshow, where the admin mutates via REST and receives
over the socket — it keeps the socket protocol small and one-directional in
practice.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import time
import uuid as _uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from backend import (context, database as db, models, prompt_book,
                     story_loader)
from backend.comfy_client import ComfyClient
from backend.gpu_broker import Consumer, GpuBroker
from backend.image_jobs import ImageJobRunner, model_name
from backend.llm_cloud import CloudClient, make_llm
from backend.session import SessionManager

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / "config.json").read_text())
MEDIA = ROOT / "data" / "images"
FRONTEND = ROOT / "frontend"

app = FastAPI(title="StoryForge")

comfy = ComfyClient(CFG)
llm = make_llm(CFG)   # cloud or local, per config.llm.provider
broker = GpuBroker(CFG, comfy, llm)
jobs: ImageJobRunner
sessions: SessionManager
_tasks: list[asyncio.Task] = []


@app.on_event("startup")
async def _startup() -> None:
    global jobs, sessions
    db.init()
    story_loader.load_all()

    async def on_job_event(kind: str, payload: dict) -> None:
        await sessions.broadcast_image_event(kind, payload)

    jobs = ImageJobRunner(CFG, broker, comfy, on_event=on_job_event)
    sessions = SessionManager(CFG, broker, llm, jobs)
    jobs.start()

    await broker.reconcile()
    requeued = await jobs.resume_pending()
    if requeued:
        print(f"  requeued {requeued} job(s) left running by a previous process")

    _tasks.append(asyncio.create_task(broker.idle_watchdog()))
    print(f"  comfy={'up' if await comfy.is_up() else 'DOWN'}  "
          f"ollama={'up' if await llm.is_up() else 'DOWN'}  "
          f"mode={CFG['gpu']['mode']}")


@app.on_event("shutdown")
async def _shutdown() -> None:
    for t in _tasks:
        t.cancel()
    with contextlib.suppress(Exception):
        await jobs.stop()
    with contextlib.suppress(Exception):
        await llm.evict()


# ---------------------------------------------------------------------- REST

@app.get("/api/stories")
async def api_stories():
    return db.list_stories()


@app.get("/api/sessions")
async def api_sessions():
    return db.list_sessions()


@app.post("/api/sessions")
async def api_create_session(story_id: str = Query(...)):
    try:
        sess = story_loader.start_session(story_id)
    except story_loader.StoryError as e:
        raise HTTPException(400, str(e))

    story = db.get_story(story_id) or {}
    opening = story.get("opening") or {}
    # The opening scene is turn 0 and is illustrated immediately, so the reader
    # has something on screen before they have typed anything.
    turn_id = db.add_turn(sess["id"], "narrator", opening.get("text", ""))
    rows = [{"kind": "prose", "text": opening.get("text", "")}]
    if opening.get("keywords"):
        from backend.turn_parser import ImageSlot
        slot = ImageSlot(index=0, entities=opening.get("entities", []),
                         keywords=opening["keywords"])
        image_id = await jobs.submit(session_id=sess["id"], turn_id=turn_id,
                                     turn_idx=0, slot=slot, style=sess["style"])
        rows.append({"kind": "image", "image_id": image_id})
    db.add_segments(turn_id, rows)
    return sess


@app.get("/api/sessions/{session_id}")
async def api_session(session_id: str):
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(404, "no such session")
    rt = sessions.get(session_id)
    return {"session": sess, **rt.replay_payload()}


@app.delete("/api/sessions/{session_id}")
async def api_delete_session(session_id: str):
    """Delete a playthrough: database rows and the images on disk."""
    sess = db.get_session(session_id)
    if not sess:
        raise HTTPException(404, "no such session")

    # Stop anything in flight first, or a running turn would write rows back
    # after the delete and a running image would land in a deleted session.
    rt = sessions.get(session_id)
    with contextlib.suppress(Exception):
        await rt.cancel_turn()
    with contextlib.suppress(Exception):
        await jobs.cancel_session(session_id)

    counts = db.delete_session(session_id)

    removed_files = 0
    folder = (MEDIA / session_id).resolve()
    if str(folder).startswith(str(MEDIA.resolve())) and folder.is_dir():
        for f in folder.iterdir():
            with contextlib.suppress(Exception):
                f.unlink()
                removed_files += 1
        with contextlib.suppress(Exception):
            folder.rmdir()

    sessions._runtimes.pop(session_id, None)
    return {"ok": bool(counts["deleted"]), "files_removed": removed_files, **counts}


@app.get("/api/stories/{story_id}/raw")
async def api_story_raw(story_id: str):
    """The full story JSON, for editing."""
    raw = db.get_story(story_id)
    if not raw:
        raise HTTPException(404, "no such story")
    return raw


@app.post("/api/stories")
async def api_save_story(payload: dict):
    """Create or overwrite a story definition.

    Writes stories/<id>.json — disk stays the source of truth — then reloads,
    so validation errors surface here rather than at the next restart.
    """
    import json as _json
    import re as _re

    sid = (payload.get("id") or "").strip().lower()
    sid = _re.sub(r"[^a-z0-9_-]+", "-", sid).strip("-")
    if not sid:
        raise HTTPException(400, "id is required (letters, numbers, - and _)")

    path = story_loader.STORIES_DIR / f"{sid}.json"
    payload["id"] = sid
    # Entities arrive as a list from the form; the file format is a map.
    ents = payload.get("entities")
    if isinstance(ents, list):
        payload["entities"] = {
            e["key"]: {"kind": e.get("kind", "character"),
                       "role": e.get("role", ""),
                       "appearance": e.get("appearance", "")}
            for e in ents if e.get("key")}

    story_loader.STORIES_DIR.mkdir(parents=True, exist_ok=True)
    previous = path.read_text() if path.exists() else None
    path.write_text(_json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    try:
        story_loader.load_all(verbose=False)
    except story_loader.StoryError as e:
        # Put the old version back rather than leaving a broken file on disk.
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            path.write_text(previous)
        raise HTTPException(400, str(e))
    return {"ok": True, "id": sid, "path": str(path)}


@app.delete("/api/stories/{story_id}")
async def api_delete_story(story_id: str, force: bool = Query(False)):
    """Remove a story definition.

    Refuses by default when playthroughs exist. `sessions.story_id` references
    `stories(id)` without a cascade, so deleting underneath them would either
    fail on the foreign key or orphan somebody's saved game. Which of those
    happens should not depend on a PRAGMA — so it is an explicit decision the
    caller has to make with ?force=true.
    """
    raw = db.get_story(story_id)
    if not raw:
        raise HTTPException(404, "no such story")

    sess_rows = db.list_sessions(story_id)
    if sess_rows and not force:
        raise HTTPException(409, f"{len(sess_rows)} playthrough(s) use this story")

    removed_sessions = 0
    removed_files = 0
    for srow in sess_rows:
        sid = srow["id"]
        rt = sessions.get(sid)
        with contextlib.suppress(Exception):
            await rt.cancel_turn()
        with contextlib.suppress(Exception):
            await jobs.cancel_session(sid)
        db.delete_session(sid)
        sessions._runtimes.pop(sid, None)
        removed_sessions += 1

        folder = (MEDIA / sid).resolve()
        if str(folder).startswith(str(MEDIA.resolve())) and folder.is_dir():
            for f in folder.iterdir():
                with contextlib.suppress(Exception):
                    f.unlink()
                    removed_files += 1
            with contextlib.suppress(Exception):
                folder.rmdir()

    db.connect().execute("DELETE FROM stories WHERE id=?", (story_id,))
    db.connect().commit()

    path = story_loader.STORIES_DIR / f"{story_id}.json"
    file_removed = False
    if path.exists() and path.resolve().parent == story_loader.STORIES_DIR.resolve():
        path.unlink()
        file_removed = True

    return {"ok": True, "id": story_id, "file_removed": file_removed,
            "sessions_removed": removed_sessions, "files_removed": removed_files}


@app.get("/api/models")
async def api_models():
    """Every checkpoint ComfyUI can load, with what prompting.toml knows."""
    return await models.catalog(CFG, comfy)


@app.post("/api/models/active")
async def api_set_model(name: str = Query(...)):
    """Switch the image model. Takes effect on the next image, everywhere.

    Validated against ComfyUI's own list: a name it cannot load would fail
    one image at a time, far from the click that caused it.
    """
    try:
        installed = await comfy.checkpoints()
        # Anima and FLUX load through the UNET loaders, so they are not in
        # the checkpoint list and were rejected as "not installed".
        installed += await comfy.diffusion_models()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"comfyui unreachable: {e}")
    if name not in installed:
        raise HTTPException(400, f"{name!r} is not installed")

    models.set_active(name)
    cat = await models.catalog(CFG, comfy)
    row = next((m for m in cat["models"] if m["name"] == name), {})
    await sessions.broadcast_all("model_changed", row)
    return {"ok": True, "active": name, "model": row}


@app.post("/api/lab/generate")
async def api_lab_generate(payload: dict):
    """Raw image generation, no story attached.

    Exists so the image side can be tuned without playing a story: give it a
    positive and negative prompt and get the picture back. Uses the same
    ComfyClient and workflow builder as the real pipeline, so what works here
    works there.
    """
    from backend import workflow as wf

    positive = (payload.get("positive") or "").strip()
    if not positive:
        raise HTTPException(400, "positive prompt is required")

    # Start from the chosen model's own settings, then let the form win.
    img_cfg = models.image_cfg(CFG, (payload.get("checkpoint") or "").strip() or None)
    for k in ("width", "height", "steps"):
        if payload.get(k):
            img_cfg[k] = int(payload[k])
    if payload.get("cfg") is not None:
        img_cfg["cfg"] = float(payload["cfg"])

    # Seed: explicit, or random. Never reuse silently — ComfyUI serves an
    # identical workflow from cache, which would look like an instant result
    # and hand back the previous picture.
    seed = int(payload["seed"]) if str(payload.get("seed", "")).strip() else \
        int.from_bytes(_uuid.uuid4().bytes[:4], "big") % (2 ** 31 - 1)

    graph = wf.build(img_cfg)
    vals = wf.values(img_cfg, prompt=positive,
                     negative=(payload.get("negative") or "").strip(),
                     seed=seed, prefix="lab")

    # The lab watches the same ComfyUI stream the stories do, so a slow model
    # looks like progress rather than a frozen button.
    job = payload.get("job_id") or _uuid.uuid4().hex

    async def on_state(s: str) -> None:
        await lab_broadcast("lab_state", {"job_id": job, "status": s})

    seen_pct = [-1]

    async def on_progress(v: int, m: int) -> None:
        # ComfyUI fires both "progress" and "progress_state" per step, so the
        # raw stream repeats every value. One frame per percent is plenty.
        pct = int(100 * v / m) if m else 0
        if pct == seen_pct[0]:
            return
        seen_pct[0] = pct
        await lab_broadcast("lab_progress", {
            "job_id": job, "value": v, "max": m, "pct": pct})

    last = [0.0]

    async def on_preview(jpeg: bytes) -> None:
        now = time.time()
        if now - last[0] < 1.0:      # same throttle the reading view uses
            return
        last[0] = now
        await lab_broadcast("lab_preview", {
            "job_id": job,
            "data_url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()})

    async with broker.lease(Consumer.IMAGE, f"lab:{job}"):
        res = await comfy.generate(wf.patch(graph, vals), on_progress=on_progress,
                                   on_preview=on_preview, on_state=on_state)
    if not res.ok or not res.images:
        await lab_broadcast("lab_failed", {"job_id": job,
                                           "error": (res.error or "")[:200]})
        raise HTTPException(500, res.error or "generation failed")

    name = f"{_uuid.uuid4().hex}.png"
    dest = MEDIA / "_lab" / name
    await comfy.download(res.images[0], dest)
    out = {"url": f"/media/_lab/{name}", "seed": seed, "job_id": job,
           "wall_s": round(res.wall_s, 1), "steps": img_cfg["steps"],
           "width": img_cfg["width"], "height": img_cfg["height"],
           "model": img_cfg.get("checkpoint", ""), "cfg": img_cfg.get("cfg")}
    await lab_broadcast("lab_done", out)
    return out


@app.get("/api/lab/defaults")
async def api_lab_defaults():
    from backend import prompt_style, ratings
    img = models.image_cfg(CFG)
    return {
        "image": {k: v for k, v in img.items() if not k.startswith("_")},
        "models": await models.catalog(CFG, comfy),
        "shapes": {k: list(v) for k, v in
                   prompt_book.resolutions(img.get("checkpoint", "")).items()},
        "negatives_by_rating": {
            r: ratings.negative_prompt(r, "", img.get("checkpoint", ""))
            for r in ratings.RATINGS},
        "positive_safety_by_rating": {r: ratings.positive_safety(r) for r in ratings.RATINGS},
        "notes": prompt_style.describe(model_name(img), img.get("arch", "")),
        "prompt_book": prompt_book.status(),
    }


@app.post("/api/sessions/{session_id}/rollback")
async def api_rollback(session_id: str, pairs: int = Query(1)):
    """Undo the last exchange so the reader can take the scene a different way."""
    if not db.get_session(session_id):
        raise HTTPException(404, "no such session")
    rt = sessions.get(session_id)
    with contextlib.suppress(Exception):
        await rt.cancel_turn()
    with contextlib.suppress(Exception):
        await jobs.cancel_session(session_id)

    removed = db.rollback_last(session_id, pairs=pairs)
    for fp in removed.pop("files", []):
        with contextlib.suppress(Exception):
            Path(fp).unlink()
    rt._last_keywords = None
    return {"ok": True, **removed, **rt.replay_payload()}


@app.get("/api/debug")
async def api_debug():
    """Broker state, queue and VRAM — the operator's view."""
    return {
        "broker": broker.snapshot(),
        "comfy_up": await comfy.is_up(),
        "ollama_up": await llm.is_up(),
        "foreign_comfy_jobs": await broker.foreign_jobs(),
        "llm_cost": llm.cost_snapshot() if isinstance(llm, CloudClient) else None,
        "pending_jobs": len(db.pending_jobs()),
        "prompt_book": prompt_book.status(),
        "active_model": models.describe(CFG, models.active(CFG)),
        "config": {"image": models.image_cfg(CFG), "llm": CFG["llm"],
                   "gpu": CFG["gpu"]},
    }


@app.get("/media/{session_id}/{name}")
async def media(session_id: str, name: str):
    p = (MEDIA / session_id / name).resolve()
    if not str(p).startswith(str(MEDIA.resolve())) or not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(p)


# ----------------------------------------------------------------- WebSocket

_lab_sockets: set[WebSocket] = set()


async def lab_broadcast(t: str, d: dict) -> None:
    """Fan a lab event out to whoever has the lab page open.

    Dead sockets are dropped rather than retried: the lab is a tuning tool,
    and a disconnected tuner has nothing to catch up on.
    """
    for ws in list(_lab_sockets):
        try:
            await ws.send_json({"t": t, "d": d})
        except Exception:  # noqa: BLE001
            _lab_sockets.discard(ws)


@app.websocket("/ws/lab")
async def ws_lab(websocket: WebSocket):
    await websocket.accept()
    _lab_sockets.add(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            if msg.get("t") == "ping":
                await websocket.send_json({"t": "pong", "d": {}})
    except WebSocketDisconnect:
        pass
    except Exception:  # noqa: BLE001
        pass
    finally:
        _lab_sockets.discard(websocket)

@app.websocket("/ws/session/{session_id}")
async def ws_session(websocket: WebSocket, session_id: str,
                     token: str = Query(default="")):
    sess = db.get_session(session_id)
    if not sess:
        await websocket.close(code=4001, reason="no such session")
        return
    if token and token != sess["token"]:
        # 4001 tells the client not to retry — gameshow's convention.
        await websocket.close(code=4001, reason="bad token")
        return

    await websocket.accept()
    rt = sessions.get(session_id)
    sock = await rt.attach(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            t = msg.get("t")
            d = msg.get("d") or {}

            if t == "action":
                text = (d.get("text") or "").strip()
                if text:
                    await rt.submit_action(text)
            elif t == "pass":
                # The reader has nothing to add and wants the world to carry on.
                await rt.submit_action(
                    "Continue. The player does nothing and simply lets time pass.",
                    passing=True)
            elif t == "retry_image":
                await jobs.retry(d.get("image_id", ""))
            elif t == "cancel_turn":
                await rt.cancel_turn()
            elif t == "ping":
                # App-level heartbeat — the piece gameshow lacks, and the thing
                # that keeps a long image generation alive behind a proxy.
                await websocket.send_json({"t": "pong", "d": {}})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        await rt.detach(sock)


# ------------------------------------------------------------------ frontend

if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
