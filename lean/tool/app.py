"""The web tool's server: the page, the design's key numbers and charts (lean.game.tuning through
lean.tool.design; the charts are PNGs drawn by the committed charts' matplotlib code), the flown
checks on demand (lean.tool.flown) and the sandbox WebSocket (lean.sandbox.world.World in a child
process)."""

import asyncio
import contextlib
import datetime
import json
import multiprocessing
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from lean.game import tuning

from . import design, session, units

PAGE = Path(__file__).with_name("index.html")
# What an unsolvable design raises in the tuning's solve (no solution, no top speed on the grid,
# a log of a non-positive number): shown to the page as the error.
UNSOLVABLE = (RuntimeError, ValueError, IndexError, ArithmeticError)


@contextlib.asynccontextmanager
async def lifespan(_):
    """At start, the preset's flown checks (about 7 s) in the background: the first charts wait
    for them."""
    threading.Thread(target=design.warm, daemon=True).start()
    yield
    design.SHUTDOWN.set()  # a solve under way ends within one evaluation


app = FastAPI(title="B&B tool", lifespan=lifespan)


class DesignRequest(BaseModel):
    """A design: the preset, its design-input profile (tuning.PROFILES; default the preset's
    own), the inputs moved (that profile's names, game units), or a saved fixed ship's card row
    (`fixed`), the time scale and the chart units."""

    preset: str = "p51d"
    profile: str | None = None
    values: dict[str, float] | None = None
    time_scale: float = tuning.TIME_FACTOR
    units: str = "game"
    fixed: dict[str, float] | None = None  # lean.tool.units.SYSTEMS (the old "real" is real_metric)


@app.exception_handler(tuning.SolveCancelled)
async def cancelled(_request, _error):
    """The server is stopping under a solve: a quiet 503."""
    return JSONResponse({"error": "cancelled"}, status_code=503)


@app.get("/")
def index():
    return FileResponse(PAGE)


SAVED = Path("/home/vizia/avesnova/boost-and-broadside/data/tool-presets")


class SavedPreset(BaseModel):
    """A named state of the Ships tab: the base preset, the inputs moved, the time scale and
    the units."""

    name: str
    state: dict


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name.strip()).strip(".")[:80]


@app.get("/api/saved")
def saved_list():
    """The saved presets (design.loaded_record: editable, or fixed ships from their stored rows);
    files it cannot read, or without a stored row, are skipped."""
    SAVED.mkdir(parents=True, exist_ok=True)
    saved = []
    for path in sorted(SAVED.glob("*.json")):
        try:
            record = design.loaded_record(json.loads(path.read_text()))
        except OSError, ValueError, AttributeError:
            continue
        if record is not None:
            saved.append(record)
    return dict(saved=saved)


@app.post("/api/saved")
def saved_save(request: SavedPreset):
    """Write the state, the table's version and the solved row to <name>.json (overwrites; the
    page confirms first)."""
    name = _safe(request.name)
    if not name:
        return JSONResponse({"error": "empty name"}, status_code=422)
    SAVED.mkdir(parents=True, exist_ok=True)
    try:
        record = design.saved_record(
            name, request.state, datetime.datetime.now().isoformat(timespec="seconds")
        )
    except UNSOLVABLE as error:
        return JSONResponse({"error": str(error)}, status_code=422)
    (SAVED / f"{name}.json").write_text(json.dumps(record, indent=2))
    return dict(name=name)


@app.delete("/api/saved/{name}")
def saved_delete(name: str):
    path = SAVED / f"{_safe(name)}.json"
    if not path.is_file():
        return JSONResponse({"error": "no such preset"}, status_code=404)
    path.unlink()
    return dict(deleted=_safe(name))


@app.get("/api/presets")
def presets():
    return dict(
        presets=design.presets(),
        profiles=design.profiles(),
        time_factor=tuning.TIME_FACTOR,
        units=dict(
            systems=units.descriptors(),
            names=units.NAMES,
            kinds=list(units.KINDS),
            order=units.SYSTEMS,
        ),
    )


def _error(error: Exception, request: DesignRequest) -> JSONResponse:
    """422 with the message; for a solve that stopped early also `failure` "
    "(design.failure_report)."""
    body = {"error": str(error)}
    if isinstance(error, tuning.SolveFailed):
        body["failure"] = design.failure_report(
            error,
            request.preset,
            request.profile,
            units.system_of(request.units),
            request.time_scale,
            sorted(request.values or {}),
        )
        body["error"] = body["failure"]["message"]
    return JSONResponse(body, status_code=422)


def _answer(compute, request: DesignRequest, png: bool):
    try:
        units.system_of(request.units)
        if png:
            return Response(
                compute(
                    request.preset,
                    request.values,
                    request.time_scale,
                    request.units,
                    fixed=request.fixed,
                    profile=request.profile,
                ),
                media_type="image/png",
            )
        return compute(
            request.preset,
            request.values,
            request.time_scale,
            fixed=request.fixed,
            profile=request.profile,
        )
    except UNSOLVABLE as error:  # the inputs have no solution
        return _error(error, request)


@app.post("/api/envelope")
def envelope(request: DesignRequest):
    """The key numbers of the design and of the preset (JSON)."""
    return _answer(design.envelope, request, png=False)


@app.post("/api/ship")
def ship(request: DesignRequest):
    """The design's implied physical quantities, warnings and solved card row (JSON)."""
    try:
        return design.ship(request.preset, request.values, request.fixed, request.profile)
    except UNSOLVABLE as error:
        return _error(error, request)


@app.websocket("/api/solve")
async def solve(websocket: WebSocket):
    """The page's solve with its progress: one DesignRequest as the first message, then
    {type: "progress", iteration, error, residuals {input: log ratio}, step, elapsed,
    tolerance} per iteration, then {type: "done", iterations, elapsed} (the design is then
    cached: the page's other requests answer at once) or {type: "failed", error, failure}.
    Closing the socket, or sending anything, cancels the solve (the latest request wins)."""
    await websocket.accept()
    loop = asyncio.get_running_loop()
    cancel = threading.Event()
    queue: asyncio.Queue = asyncio.Queue()
    try:
        request = DesignRequest(**await websocket.receive_json())
        units.system_of(request.units)
    except WebSocketDisconnect:
        return
    except (ValueError, TypeError) as error:
        await websocket.send_json({"type": "failed", "error": str(error), "failure": None})
        await websocket.close()
        return

    def progress(step):
        loop.call_soon_threadsafe(
            queue.put_nowait,
            dict(
                type="progress",
                iteration=step.iteration,
                error=step.worst,
                residuals=dict(zip(step.names, step.residuals)),
                step=step.step,
                elapsed=step.elapsed,
                tolerance=step.tolerance,
            ),
        )

    def work():
        try:
            return design.solve(
                request.preset,
                request.values,
                request.fixed,
                request.profile,
                progress=progress,
                cancel=cancel,
            )
        except tuning.SolveCancelled:
            return None
        except UNSOLVABLE as error:
            return error

    began = loop.time()
    solving = asyncio.ensure_future(run_in_threadpool(work))
    listening = asyncio.ensure_future(websocket.receive_text())
    try:
        while True:
            getting = asyncio.ensure_future(queue.get())
            await asyncio.wait({getting, solving, listening}, return_when=asyncio.FIRST_COMPLETED)
            if listening.done():  # a new message or the page went: this solve is no longer wanted
                cancel.set()
                getting.cancel()
                await asyncio.wait({solving})
                return
            if getting.done():
                await websocket.send_json(getting.result())
                continue
            getting.cancel()
            while not queue.empty():
                await websocket.send_json(queue.get_nowait())
            result = solving.result()
            if isinstance(result, Exception):
                message = _error(result, request)
                await websocket.send_json({"type": "failed", **json.loads(message.body)})
            else:
                await websocket.send_json(
                    dict(
                        type="done",
                        elapsed=loop.time() - began,
                        iterations=0 if result is None else result.iterations,
                    )
                )
            return
    except WebSocketDisconnect:
        cancel.set()
    finally:
        cancel.set()
        listening.cancel()
        with contextlib.suppress(RuntimeError):
            await websocket.close()


@app.post("/api/doghouse.png")
def doghouse(request: DesignRequest):
    """The design's doghouse drawn by the committed charts' matplotlib code."""
    return _answer(design.doghouse_png, request, png=True)


@app.post("/api/sustained.png")
def sustained(request: DesignRequest):
    """The design's sustained turn against speed, the preset's dotted (matplotlib)."""
    return _answer(design.sustained_png, request, png=True)


@app.post("/api/flown")
def flown_checks(request: DesignRequest):
    """Fly the design's checks in the game (seconds): the lowest speed of each line and the key
    points; the charts then draw them."""
    return _answer(design.fly_checks, request, png=False)


def _receive(connection, process):
    """Block (in a thread) until the child sends a message; None once it has gone."""
    while process.is_alive() or connection.poll():
        if connection.poll(0.5):
            return connection.recv()
    return None


@app.websocket("/api/sandbox")
async def sandbox(websocket: WebSocket):
    """The first message {preset, profile?, values, fixed?, time_scale, opponent, ticks?} starts
    a session; then {side, level, power, fire} (the held controls), {reset: true} and
    {opponent: bool} go to the World, and every tick comes back as a frame
    (lean.tool.session.frame)."""
    await websocket.accept()
    process = None
    try:
        start = await websocket.receive_json()
        try:
            row = await run_in_threadpool(
                design.build_row,
                start.get("preset", "p51d"),
                start.get("values"),
                start.get("fixed"),
                start.get("profile"),
            )
        except UNSOLVABLE as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=session.run,
            daemon=True,
            args=(
                child,
                row,
                float(start.get("time_scale", tuning.TIME_FACTOR)),
                bool(start.get("opponent", True)),
                start.get("ticks") is None,
                start.get("ticks"),
            ),
        )
        process.start()
        loop = asyncio.get_running_loop()

        async def forward():
            while (
                message := await loop.run_in_executor(None, _receive, parent, process)
            ) is not None:
                await websocket.send_json(message)

        sender = asyncio.create_task(forward())
        try:
            while not sender.done():
                receiving = asyncio.ensure_future(websocket.receive_json())
                await asyncio.wait({receiving, sender}, return_when=asyncio.FIRST_COMPLETED)
                if not receiving.done():
                    receiving.cancel()
                    break
                parent.send(receiving.result())
        finally:
            sender.cancel()
    except WebSocketDisconnect:
        pass
    finally:
        if process is not None:
            process.terminate()
            process.join(timeout=5)
        try:
            await websocket.close()
        except RuntimeError:
            pass
