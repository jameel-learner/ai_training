"""Notebook 3 — Production coding agent (Agent Mode).

plan (codebase RAG + structured Plan) -> code (sequential, or parallel via Send)
-> AI review (auto-approve after 2 rejections) -> human review (interrupt)
-> apply -> test.  MemorySaver checkpoints every step (time travel via history()).
"""

from __future__ import annotations

import difflib
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, AsyncGenerator

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from . import codebase
from .common import WORKSPACE, _env, get_llm, list_files, load_rules, rel, resolve, structured, write_text

MAX_AI_REJECTIONS = 2
SCRIPT_MARKERS = ("__main__", "streamlit", "gradio", "app.run(", "uvicorn.run(", "input(", "mainloop(")
RESET = "__reset__"


# ---------------------------------------------------------------- schemas
class FileTask(BaseModel):
    filepath: str = Field(description="Path of the file to create or modify, relative to the workspace root")
    description: str = Field(description="What to do with this file")
    action: str = Field(description="'create' or 'modify'")


class Plan(BaseModel):
    summary: str = Field(description="One-line summary of the plan")
    file_tasks: list[FileTask] = Field(description="List of file-level tasks")


class CodeResult(BaseModel):
    code: str = Field(description="Complete file content")
    explanation: str = Field(description="Brief explanation of changes")




class ReviewFeedback(BaseModel):
    approved: bool = Field(description="Whether the code changes are approved")
    feedback: str = Field(description="Brief feedback on the code")


# ---------------------------------------------------------------- state
def merge_code(existing: list | None, new: list | None) -> list:
    """Reducer: parallel coders append; a leading RESET marker starts a fresh set."""
    existing = existing or []
    new = new or []
    if new and new[0] == RESET:
        return list(new[1:])
    return existing + new


class OrchestratorState(TypedDict, total=False):
    feature_request: str
    project: str
    model: str
    parallel: bool
    rules: str
    codebase_context: str
    plan: str
    file_tasks: list[dict]
    generated_code: Annotated[list[dict], merge_code]
    review_result: str
    review_attempts: int
    human_decision: str
    human_feedback: str
    test_output: str
    status: str


class SingleFileState(TypedDict):
    task: dict
    feature_request: str
    plan: str
    codebase_context: str
    model: str
    feedback: str
    all_tasks: list[dict]
    rules: str


# ---------------------------------------------------------------- helpers
def _normalize(filepath: str, project: str) -> str:
    p = Path(filepath.replace("\\", "/").lstrip("/"))
    if p.parts and p.parts[0] == "workspace":
        p = Path(*p.parts[1:])
    if project and project != "." and (not p.parts or p.parts[0] != project):
        p = Path(project) / p
    return rel(resolve(p.as_posix()))


def _feedback(state: OrchestratorState) -> str:
    parts = []
    if state.get("human_decision") == "reject" and state.get("human_feedback"):
        parts.append(f"The human reviewer rejected the previous version: {state['human_feedback']}")
    elif state.get("human_decision") == "reject":
        parts.append("The human reviewer rejected the previous version; take a different, better approach.")
    if state.get("status") == "needs_revision" and state.get("review_result"):
        parts.append(f"The AI reviewer asked for changes: {state['review_result']}")
    return "\n".join(parts)


BLOCK_RE = re.compile(
    r"^<{5,9} ?SEARCH[ \t]*\n(.*?)^={5,9}[ \t]*\n(.*?)^>{5,9} ?REPLACE[ \t]*$", re.S | re.M
)
EDIT_FORMAT = """Reply in exactly this format and nothing else:

EXPLANATION: <one line describing the change>

<<<<<<< SEARCH
<exact lines copied from the current file>
=======
<the new lines that replace them>
>>>>>>> REPLACE

Use one block per change (as many blocks as needed). Rules for blocks:
- SEARCH must be copied character-for-character from the current file, including indentation,
  and must match exactly one place. Keep it short: the lines you change plus a line of context.
- To insert code, SEARCH an existing anchor line and REPLACE it with that same line plus the new lines.
- Write real code with real newlines and correct indentation — no escape sequences, no markdown fences.
"""


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _parse_edits(text: str) -> tuple[list[tuple[str, str]], str]:
    m = re.search(r"EXPLANATION:\s*(.+)", text)
    return BLOCK_RE.findall(text), (m.group(1).strip() if m else "")


def _apply_edits(text: str, edits: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Apply search/replace edits; every SEARCH block must match exactly once."""
    if not edits:
        return text, ["no SEARCH/REPLACE blocks found in your reply"]
    errors = []
    for i, (search, replace) in enumerate(edits, 1):
        if not search.strip():
            errors.append(f"block {i}: empty SEARCH")
            continue
        count = text.count(search)
        if count == 0:  # tolerate trailing-whitespace differences
            s2 = "\n".join(l.rstrip() for l in search.split("\n"))
            t2 = "\n".join(l.rstrip() for l in text.split("\n"))
            if t2.count(s2) == 1:
                text, search, count = t2, s2, 1
        if count != 1:
            errors.append(f"block {i}: SEARCH {'not found' if count == 0 else f'matches {count} places'}:\n{search[:300]}")
            continue
        text = text.replace(search, replace, 1)
    return text, errors


def _syntax_error(code: str, filepath: str) -> str:
    if not filepath.endswith(".py"):
        return ""
    try:
        compile(code, filepath, "exec")
        return ""
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} at line {e.lineno}: {(e.text or '').rstrip()}"


def _generate_file(
    task: dict, feature: str, plan: str, context: str, model: str, feedback: str,
    all_tasks: list[dict] | None = None, done: list[dict] | None = None, rules: str = "",
) -> dict:
    existing = ""
    if task["action"] == "modify":
        try:
            existing = resolve(task["filepath"]).read_text(encoding="utf-8").replace("\r\n", "\n")
        except FileNotFoundError:
            pass
    if existing and not existing.endswith("\n"):
        existing += "\n"

    others = [t for t in (all_tasks or []) if t["filepath"] != task["filepath"]]
    header = (
        f"Project rules (MUST follow — they override your own assumptions about APIs, URLs and models):\n{rules}\n\n"
        if rules else ""
    )
    header += f"Overall feature request: {feature}\nPlan: {plan}\n"
    if others:
        header += (
            "Other files are being changed at the same time — assume these changes exist and stay consistent "
            "with them (e.g. pass new parameters / call new functions they add):\n"
            + "\n".join(f"- {t['filepath']}: {t['description']}" for t in others)
            + "\n"
        )
    header += f"\nYOUR task: {task['description']}\nFile: {task['filepath']}\n"
    header += (
        "Note: paths like the one above are only how this tool locates files. Inside the code, never hardcode them "
        "— refer to the project's own files relative to the source file (e.g. Path(__file__).parent / 'data.json'), "
        "and use plain module imports between project files.\n"
    )
    tail = ""
    if context:
        tail += f"\nRelated codebase context:\n{context[:3000]}\n"
    for d in done or []:
        tail += f"\nAlready written in this change — {d['filepath']} (stay consistent with it):\n```\n{d['code'][:4000]}\n```\n"
    if feedback:
        tail += f"\nFeedback on the previous attempt — address it:\n{feedback}\n"

    llm = get_llm(model)
    problem, code, explanation = "", "", ""
    for _ in range(3):  # first try + up to 2 repair rounds
        if existing:
            # Modify: minimal SEARCH/REPLACE edits so untouched code is preserved byte-for-byte.
            prompt = (
                "You are editing an existing, WORKING file. Make the smallest change that implements YOUR task.\n"
                "- Never rewrite, reorder or reformat code unrelated to the task.\n"
                "- Do not change how existing functions are called or how their return values are used "
                "(e.g. if a function's result is iterated, awaited or passed to another call, keep it that way) "
                "unless the task requires it.\n"
                "- Keep existing imports, names and behaviour; put new code next to related code, "
                "after anything it depends on (a variable or state must be initialised before it is used).\n"
                "- Follow this file's existing style, libraries and conventions.\n"
                "- Never mutate arguments passed into a function (copy lists before changing them).\n\n"
                f"{header}\nCurrent file ({task['filepath']}):\n```\n{existing}```\n{tail}\n{EDIT_FORMAT}"
            )
            if problem:
                prompt += f"\n\nYour previous reply could not be used:\n{problem}\nReply again with ALL blocks, fixed."
            edits, explanation = _parse_edits(_text(llm.invoke(prompt).content))
            code, errors = _apply_edits(existing, edits)
            if errors:
                problem = "\n\n".join(errors)
                continue
        else:
            prompt = (
                "Generate the COMPLETE content for this new file.\n\n" + header + tail
                + "\nReply in exactly this format:\nEXPLANATION: <one line>\n\n```\n<entire file content>\n```"
            )
            if problem:
                prompt += f"\n\nYour previous version:\n```\n{code}\n```\nfailed: {problem}\nFix it."
            reply = _text(llm.invoke(prompt).content)
            fences = re.findall(r"^```[\w.+-]*[ \t]*\n(.*?)^```[ \t]*$", reply, re.S | re.M)
            if not fences:
                problem = "no fenced code block found in your reply"
                continue
            code = max(fences, key=len)
            m = re.search(r"EXPLANATION:\s*(.+)", reply)
            explanation = m.group(1).strip() if m else ""
        problem = _syntax_error(code, task["filepath"])
        if not problem:
            break
        if existing:
            problem += f"\nThe file after applying your edits was:\n```\n{code}```"
    if problem:
        raise RuntimeError(f"Coder could not produce valid code for {task['filepath']}: {problem[:400]}")

    return {
        "filepath": task["filepath"],
        "action": task["action"],
        "code": code,
        "explanation": explanation or task["description"],
    }


# ---------------------------------------------------------------- nodes
def plan_node(state: OrchestratorState) -> OrchestratorState:
    project = state.get("project") or "."
    docs, method = codebase.search(state["feature_request"], project, k=4)
    context = codebase.format_results(docs)
    files = list_files(project)
    plan: Plan = structured(state.get("model"), Plan).invoke(
        "You are a coding planner. Create a minimal plan for this feature: which files to create or modify and why.\n"
        f"All file paths must be relative to the workspace root and live under '{project}/'.\n"
        "Work with the project's EXISTING files and structure as listed below: prefer modifying existing files, "
        "and only create a new file when the feature genuinely needs one. If the project is empty, choose a "
        "simple structure that fits the request.\n\n"
        + (f"Project rules (must follow):\n{state['rules']}\n\n" if state.get("rules") else "")
        + f"Feature: {state['feature_request']}\n\n"
        f"Files in project:\n" + "\n".join(files[:200]) + "\n\n"
        f"Relevant codebase context (retrieved via {method} search):\n{context}"
    )
    tasks = []
    for ft in plan.file_tasks:
        t = ft.model_dump()
        t["filepath"] = _normalize(t["filepath"], project)
        t["action"] = "modify" if resolve(t["filepath"]).exists() else "create"
        tasks.append(t)
    return {
        "codebase_context": context,
        "plan": plan.summary,
        "file_tasks": tasks,
        "status": "planned",
        "review_attempts": 0,
    }


def dispatch_node(state: OrchestratorState) -> OrchestratorState:
    """Clears the previous coding pass before (re)coding."""
    return {"generated_code": [RESET]}


def route_dispatch(state: OrchestratorState):
    if state.get("parallel") and state.get("file_tasks"):
        fb = _feedback(state)
        return [
            Send(
                "code_file",
                {
                    "task": t,
                    "feature_request": state["feature_request"],
                    "plan": state.get("plan", ""),
                    "codebase_context": state.get("codebase_context", ""),
                    "model": state.get("model"),
                    "feedback": fb,
                    "rules": state.get("rules", ""),
                    "all_tasks": state["file_tasks"],
                },
            )
            for t in state["file_tasks"]
        ]
    return "code"


def code_node(state: OrchestratorState) -> OrchestratorState:
    fb = _feedback(state)
    results: list[dict] = []
    for t in state.get("file_tasks", []):
        # sequential mode: each coder sees the files already written in this pass, so formats/APIs line up
        results.append(
            _generate_file(t, state["feature_request"], state.get("plan", ""), state.get("codebase_context", ""),
                           state.get("model"), fb, state.get("file_tasks", []), done=results,
                           rules=state.get("rules", ""))
        )
    return {"generated_code": results, "status": "coded"}


def code_file_node(state: SingleFileState) -> OrchestratorState:
    item = _generate_file(state["task"], state["feature_request"], state["plan"], state["codebase_context"], state["model"], state["feedback"], state["all_tasks"], rules=state.get("rules", ""))
    return {"generated_code": [item]}


def collect_node(state: OrchestratorState) -> OrchestratorState:
    return {"status": "coded"}


def review_node(state: OrchestratorState) -> OrchestratorState:
    attempts = state.get("review_attempts", 0)
    if attempts >= MAX_AI_REJECTIONS:
        return {
            "review_result": f"auto-approved after {attempts} review cycles — forwarding to human",
            "review_attempts": attempts + 1,
            "status": "approved",
        }
    if not state.get("generated_code"):
        return {"review_result": "no changes generated", "review_attempts": attempts + 1, "status": "approved"}
    summary = "\n".join(f"--- {i['filepath']} ---\n{i['code'][:4000]}" for i in state["generated_code"])
    result: ReviewFeedback = structured(state.get("model"), ReviewFeedback).invoke(
        f"Review these code changes for the feature request: {state['feature_request']}\n"
        "Approve if the code is functional and correct. Only reject for serious bugs, wrong functionality, "
        "or violations of the project rules below — including any external API URL, endpoint path, auth or "
        "model id that contradicts them, and any hardcoded secret.\n\n"
        + (f"Project rules:\n{state['rules']}\n\n" if state.get("rules") else "")
        + summary
    )
    return {
        "review_result": result.feedback,
        "review_attempts": attempts + 1,
        "status": "approved" if result.approved else "needs_revision",
    }


def route_after_review(state: OrchestratorState) -> str:
    return "human_review" if state["status"] == "approved" else "dispatch"


def human_review_node(state: OrchestratorState) -> OrchestratorState:
    files = []
    for item in state.get("generated_code", []):
        path = resolve(item["filepath"])
        old = path.read_text(encoding="utf-8") if path.exists() else ""
        diff = "".join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                item["code"].splitlines(keepends=True),
                fromfile=f"a/{item['filepath']}" if old else "/dev/null",
                tofile=f"b/{item['filepath']}",
            )
        )
        files.append({**item, "diff": diff})
    decision = interrupt(
        {
            "plan": state.get("plan", ""),
            "review_result": state.get("review_result", ""),
            "review_attempts": state.get("review_attempts", 0),
            "files": files,
        }
    )
    if isinstance(decision, str):
        decision = {"decision": decision}
    d = decision.get("decision", "reject")
    out: OrchestratorState = {"human_decision": d, "human_feedback": decision.get("feedback", ""), "status": "human_reviewed"}
    if d != "approve":
        out["review_attempts"] = 0  # give the AI reviewer a fresh budget for the new attempt
    return out


def route_after_human(state: OrchestratorState) -> str:
    return "apply" if state.get("human_decision") == "approve" else "dispatch"


def apply_node(state: OrchestratorState) -> OrchestratorState:
    for item in state.get("generated_code", []):
        write_text(item["filepath"], item["code"])
    return {"status": "applied"}


def test_node(state: OrchestratorState) -> OrchestratorState:
    """Compile every applied .py file, then import the non-UI modules (like NB3's import check)."""
    lines, ok = [], True
    for item in state.get("generated_code", []):
        if not item["filepath"].endswith(".py"):
            continue
        path = resolve(item["filepath"])
        r = subprocess.run([sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True, env=_env())
        good = r.returncode == 0
        ok &= good
        lines.append(f"{'PASS' if good else 'FAIL'}  compile {item['filepath']}" + ("" if good else f"\n{r.stderr.strip()}"))
        # import-check library modules only; entry-point/UI/server scripts would start running when imported
        if good and not any(m in item["code"] for m in SCRIPT_MARKERS):
            r = subprocess.run(
                [sys.executable, "-c", f"import {path.stem}"],
                cwd=path.parent, capture_output=True, text=True, timeout=20, env=_env(),
            )
            good = r.returncode == 0
            ok &= good
            lines.append(f"{'PASS' if good else 'FAIL'}  import  {path.stem}" + ("" if good else f"\n{r.stderr.strip()[-800:]}"))
    if not lines:
        lines.append("No Python files to test.")
    return {"test_output": "\n".join(lines), "status": "tests_passed" if ok else "tests_failed"}


# ---------------------------------------------------------------- graph
_g = StateGraph(OrchestratorState)
_g.add_node("plan", plan_node)
_g.add_node("dispatch", dispatch_node)
_g.add_node("code", code_node)
_g.add_node("code_file", code_file_node)
_g.add_node("collect", collect_node)
_g.add_node("review", review_node)
_g.add_node("human_review", human_review_node)
_g.add_node("apply", apply_node)
_g.add_node("test", test_node)
_g.add_edge(START, "plan")
_g.add_edge("plan", "dispatch")
_g.add_conditional_edges("dispatch", route_dispatch, ["code", "code_file"])
_g.add_edge("code", "review")
_g.add_edge("code_file", "collect")
_g.add_edge("collect", "review")
_g.add_conditional_edges("review", route_after_review, {"human_review": "human_review", "dispatch": "dispatch"})
_g.add_conditional_edges("human_review", route_after_human, {"apply": "apply", "dispatch": "dispatch"})
_g.add_edge("apply", "test")
_g.add_edge("test", END)
graph = _g.compile(checkpointer=MemorySaver())


def _clean(out) -> dict:
    if not isinstance(out, dict):
        return {}
    out = dict(out)
    if "generated_code" in out:
        out["generated_code"] = [
            {k: v for k, v in i.items() if k != "code"} | {"lines": i["code"].count("\n") + 1}
            for i in out["generated_code"]
            if isinstance(i, dict)
        ]
    out.pop("codebase_context", None)
    return out


async def _run(inp, thread_id: str) -> AsyncGenerator[dict, None]:
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}
    async for chunk in graph.astream(inp, config, stream_mode="updates"):
        for node, out in chunk.items():
            if node == "__interrupt__":
                yield {"type": "interrupt", "payload": out[0].value}
            else:
                yield {"type": "node", "node": node, "data": _clean(out)}
    snap = graph.get_state(config)
    if not snap.next:
        v = snap.values
        yield {"type": "done", "status": v.get("status"), "test_output": v.get("test_output", ""),
               "files": [i["filepath"] for i in v.get("generated_code", [])]}


def start(feature_request: str, project: str, parallel: bool, model: str | None, thread_id: str, rules: str = ""):
    project = rel(resolve(project or "."))
    return _run(
        {
            "feature_request": feature_request,
            "project": project,
            "parallel": parallel,
            "model": model,
            "rules": load_rules(rules),
            "generated_code": [RESET],
            "review_attempts": 0,
            "human_decision": "",
            "human_feedback": "",
        },
        thread_id,
    )


def resume(thread_id: str, decision: str, feedback: str):
    return _run(Command(resume={"decision": decision, "feedback": feedback}), thread_id)


def history(thread_id: str) -> list[dict]:
    """Time travel: every checkpoint of a thread, oldest first."""
    config = {"configurable": {"thread_id": thread_id}}
    out = []
    for i, snap in enumerate(reversed(list(graph.get_state_history(config)))):
        v = snap.values
        out.append(
            {
                "step": i,
                "status": v.get("status", "initial"),
                "next": list(snap.next),
                "files": len(v.get("generated_code", []) or []),
                "review_attempts": v.get("review_attempts", 0),
                "plan": (v.get("plan") or "")[:80],
                "created_at": snap.created_at,
            }
        )
    return out


__all__ = ["start", "resume", "history", "WORKSPACE"]
