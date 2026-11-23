"""The web tool's server: the page, the envelope endpoint (lean.game.tuning through
lean.tool.design) and the sandbox WebSocket (lean.sandbox.world.World in a child process)."""

import asyncio
import multiprocessing
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from lean.game import tuning

from . import design, session

PAGE = Path(__file__).with_name("index.html")
app = FastAPI(title="B&B tool")


class EnvelopeRequest(BaseModel):
    preset: str = "p51d"
    handles: dict[str, float] | None = None
    time_scale: float = tuning.TIME_FACTOR


@app.get("/")
def index():
    return FileResponse(PAGE)


@app.get("/api/presets")
def presets():
    return dict(presets=design.presets(), time_factor=tuning.TIME_FACTOR)


@app.post("/api/envelope")
def envelope(request: EnvelopeRequest):
    try:
        return design.envelope(request.preset, request.handles, request.time_scale)
    except (RuntimeError, ValueError) as error:  # the handles have no solution
        return JSONResponse({"error": str(error)}, status_code=422)


def _receive(connection, process):
    """Block (in a thread) until the child sends a message; None once it has gone."""
    while process.is_alive() or connection.poll():
        if connection.poll(0.5):
            return connection.recv()
    return None


@app.websocket("/api/sandbox")
async def sandbox(websocket: WebSocket):
    """The first message {preset, handles, time_scale, opponent, ticks?} starts a session; then
    {side, level, power, fire} (the held controls), {reset: true} and {opponent: bool} go to the
    World, and every tick comes back as a frame (lean.tool.session.frame)."""
    await websocket.accept()
    process = None
    try:
        start = await websocket.receive_json()
        try:
            row = await run_in_threadpool(design.build_row, start.get("preset", "p51d"), start.get("handles"))
        except (RuntimeError, ValueError) as error:
            await websocket.send_json({"type": "error", "message": str(error)})
            return
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=session.run, daemon=True,
            args=(child, row, float(start.get("time_scale", tuning.TIME_FACTOR)),
                  bool(start.get("opponent", True)), start.get("ticks") is None, start.get("ticks")),
        )
        process.start()
        loop = asyncio.get_running_loop()

        async def forward():
            while (message := await loop.run_in_executor(None, _receive, parent, process)) is not None:
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
