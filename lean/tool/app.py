"""The web tool's server: the page, the design's key numbers and charts (lean.game.tuning through
lean.tool.design; the charts are PNGs drawn by the committed charts' matplotlib code), the flown
checks on demand (lean.tool.flown) and the sandbox WebSocket (lean.sandbox.world.World in a child
process)."""

import asyncio
import contextlib
import multiprocessing
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from lean.game import tuning

from . import design, session

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


app = FastAPI(title="B&B tool", lifespan=lifespan)


class DesignRequest(BaseModel):
    """A design: the preset, the inputs changed (design.inputs names), the switches off (None:
    the preset's), the time scale and the chart units."""

    preset: str = "p51d"
    values: dict[str, float] | None = None
    off: list[str] | None = None
    time_scale: float = tuning.TIME_FACTOR
    units: str = "game"


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/api/presets")
def presets():
    return dict(presets=design.presets(), time_factor=tuning.TIME_FACTOR)


def _answer(compute, request: DesignRequest, png: bool):
    try:
        if png:
            return Response(
                compute(
                    request.preset,
                    request.values,
                    request.off,
                    request.time_scale,
                    request.units == "real",
                ),
                media_type="image/png",
            )
        return compute(request.preset, request.values, request.off, request.time_scale)
    except UNSOLVABLE as error:  # the inputs have no solution
        return JSONResponse({"error": str(error)}, status_code=422)


@app.post("/api/envelope")
def envelope(request: DesignRequest):
    """The key numbers of the design and of the preset (JSON)."""
    return _answer(design.envelope, request, png=False)


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
    """The first message {preset, values, off, time_scale, opponent, ticks?} starts a session; then
    {side, level, power, fire} (the held controls), {reset: true} and {opponent: bool} go to the
    World, and every tick comes back as a frame (lean.tool.session.frame)."""
    await websocket.accept()
    process = None
    try:
        start = await websocket.receive_json()
        try:
            row = await run_in_threadpool(
                design.build_row, start.get("preset", "p51d"), start.get("values"), start.get("off")
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
