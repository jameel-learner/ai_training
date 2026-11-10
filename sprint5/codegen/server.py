"""Codegen Studio — one web app for all three codegen notebooks.

  chat   (NB1)  tool-calling code generator with streaming + multi-turn memory
  bugbot (NB2)  generate -> execute -> review self-correcting loop, rules, inline edit
  agent  (NB3)  plan -> code (seq/parallel) -> review -> human approval -> apply -> test

Run:  python server.py   then open http://127.0.0.1:8765
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agents import bugbot, chat_agent, orchestrator
from agents.common import (
    BASE_DIR,
    DEFAULT_MODEL,
    OPENROUTER_API_KEY,
    WORKSPACE,
    ensure_workspace,
    list_files,
    read_text,
    rel,
    reset_sample_project,
    write_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
ensure_workspace()

app = FastAPI(title="Codegen Studio")


def sse(gen: AsyncIterator[dict]) -> StreamingResponse:
    async def body():
        try:
            async for event in gen:
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except Exception as e:  # noqa: BLE001
            logging.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'message': f'{type(e).__name__}: {e}'})}\n\n"

    return StreamingResponse(body(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- models
class ChatReq(BaseModel):
    message: str
    thread_id: str
    model: str | None = None
    rules: str = ""
    allow_shell: bool = False


class BugbotReq(BaseModel):
    task: str
    existing_code: str = ""
    rules: str = ""
    max_attempts: int = 3
    review: bool = True
    model: str | None = None


class AgentStartReq(BaseModel):
    feature_request: str
    thread_id: str
    project: str = "sample_project"
    parallel: bool = False
    model: str | None = None
    rules: str = ""


class AgentResumeReq(BaseModel):
    thread_id: str
    decision: str  # approve | reject
    feedback: str = ""


# ---------------------------------------------------------------- routes
@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/config")
def config():
    dirs = sorted({f.split("/")[0] for f in list_files(".") if "/" in f})
    return {"model": DEFAULT_MODEL, "has_key": bool(OPENROUTER_API_KEY), "workspace": str(WORKSPACE), "projects": dirs}


@app.get("/api/files")
def files():
    return {"files": list_files(".")}


@app.get("/api/file")
def file(path: str):
    try:
        return {"path": path, "content": read_text(path)}
    except (ValueError, FileNotFoundError, IsADirectoryError, UnicodeDecodeError) as e:
        raise HTTPException(400, str(e)) from e


class SaveReq(BaseModel):
    path: str
    content: str


@app.post("/api/save")
def save(req: SaveReq):
    try:
        return {"path": rel(write_text(req.path, req.content))}
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.post("/api/chat")
def chat(req: ChatReq):
    return sse(chat_agent.stream_chat(req.message, req.thread_id, req.model, req.rules, req.allow_shell))


@app.post("/api/bugbot")
def run_bugbot(req: BugbotReq):
    return sse(bugbot.stream_bugbot(req.task, req.existing_code, req.rules, req.max_attempts, req.review, req.model))


@app.post("/api/agent/start")
def agent_start(req: AgentStartReq):
    return sse(orchestrator.start(req.feature_request, req.project, req.parallel, req.model, req.thread_id, req.rules))


@app.post("/api/agent/resume")
def agent_resume(req: AgentResumeReq):
    return sse(orchestrator.resume(req.thread_id, req.decision, req.feedback))


@app.get("/api/agent/history/{thread_id}")
def agent_history(thread_id: str):
    return {"history": orchestrator.history(thread_id)}


@app.post("/api/workspace/reset-sample")
def reset_sample():
    reset_sample_project()
    return {"ok": True}


if __name__ == "__main__":
    # Bound to localhost only: the agents can write files and run code/shell commands.
    uvicorn.run(app, host="127.0.0.1", port=8765)
