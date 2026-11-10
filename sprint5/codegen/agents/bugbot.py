"""Notebook 2 — Self-correcting code agent (Bugbot).

Generate -> Execute -> Review with bounded retries, structured output,
dynamic rules injection and inline edits of existing code.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from .common import execute_python, load_rules, structured


class CodeOutput(BaseModel):
    code: str = Field(description="The complete, self-contained Python code to execute")
    explanation: str = Field(description="Brief explanation of what the code does")


class ReviewResult(BaseModel):
    approved: bool = Field(description="Whether the code meets quality standards")
    feedback: str = Field(description="Specific feedback for improvement if not approved")


class BugbotState(TypedDict, total=False):
    task: str
    existing_code: str  # inline edit (Cmd+K) target
    rules: str  # custom coding rules (.cursorrules)
    model: str
    review: bool  # run the reflection/reviewer node
    code: str
    explanation: str
    execution_result: str
    error: str
    review_feedback: str
    attempts: int
    max_attempts: int
    status: str


GEN_GUIDE = (
    "Return complete, self-contained Python that runs with `python script.py`: "
    "no input(), no network calls, standard library unless the task says otherwise, "
    "and it must print its results (include a small test/demo under `if __name__ == '__main__':`)."
)


def generate(state: BugbotState) -> BugbotState:
    if state.get("existing_code"):
        prompt = (
            f"Modify this existing code:\n```python\n{state['existing_code']}\n```\n"
            f"Changes requested:\n{state['task']}"
        )
    else:
        prompt = f"Write Python code for: {state['task']}"

    if state.get("rules"):
        prompt = f"Follow these rules:\n{state['rules']}\n\n{prompt}"

    if state.get("error"):
        prompt += (
            f"\n\nYour previous attempt:\n```python\n{state.get('code', '')}\n```\n"
            f"failed with an execution error:\n{state['error']}\n\nFix the code."
        )
    elif state.get("review_feedback"):
        prompt += (
            f"\n\nYour previous code:\n```python\n{state.get('code', '')}\n```\n"
            f"worked but the reviewer said:\n{state['review_feedback']}\n\nImprove the code."
        )

    result: CodeOutput = structured(state.get("model"), CodeOutput).invoke(f"{GEN_GUIDE}\n\n{prompt}")
    return {
        "code": result.code,
        "explanation": result.explanation,
        "attempts": state.get("attempts", 0) + 1,
        "status": "executing",
        "error": "",
        "review_feedback": "",
    }


def execute(state: BugbotState) -> BugbotState:
    r = execute_python(state["code"])
    if r["returncode"] == 0:
        return {"execution_result": r["stdout"], "error": "", "status": "executed"}
    return {"execution_result": r["stdout"], "error": r["stderr"] or f"exit code {r['returncode']}", "status": "exec_failed"}


def review(state: BugbotState) -> BugbotState:
    rules = f"\n\nThe code must also follow these rules:\n{state['rules']}" if state.get("rules") else ""
    result: ReviewResult = structured(state.get("model"), ReviewResult).invoke(
        "Review this Python code for quality (type hints, naming, PEP 8, efficiency, correctness). "
        "Set approved=true if it correctly solves the task, runs, and is reasonably clean (and follows the rules, if any). "
        "Only set approved=false for real bugs, wrong behaviour, or clear rule violations — not for minor style nitpicks; "
        "put optional suggestions in feedback.\n\n"
        f"Task: {state['task']}{rules}\n\n```python\n{state['code']}\n```\n\n"
        f"Execution output:\n{state.get('execution_result', '')[:1500]}"
    )
    if result.approved:
        return {"status": "approved", "review_feedback": result.feedback}
    return {"status": "review_failed", "review_feedback": result.feedback}


def route_after_execute(state: BugbotState) -> str:
    if state["status"] == "executed":
        return "review" if state.get("review", True) else "done"
    if state["attempts"] < state.get("max_attempts", 3):
        return "retry"
    return "done"


def route_after_review(state: BugbotState) -> str:
    if state["status"] == "approved":
        return "done"
    if state["attempts"] < state.get("max_attempts", 3):
        return "revise"
    return "done"


_g = StateGraph(BugbotState)
_g.add_node("generate", generate)
_g.add_node("execute", execute)
_g.add_node("review", review)
_g.add_edge(START, "generate")
_g.add_edge("generate", "execute")
_g.add_conditional_edges("execute", route_after_execute, {"review": "review", "retry": "generate", "done": END})
_g.add_conditional_edges("review", route_after_review, {"done": END, "revise": "generate"})
graph = _g.compile()


async def stream_bugbot(
    task: str, existing_code: str, rules: str, max_attempts: int, review_enabled: bool, model: str | None
) -> AsyncGenerator[dict, None]:
    state: dict = {
        "task": task,
        "existing_code": existing_code,
        "rules": load_rules(rules),
        "model": model,
        "review": review_enabled,
        "attempts": 0,
        "max_attempts": max(1, min(max_attempts, 6)),
    }
    final = dict(state)
    async for chunk in graph.astream(state, stream_mode="updates"):
        for node, out in chunk.items():
            out = out or {}
            final.update(out)
            yield {"type": "node", "node": node, "data": out, "attempts": final.get("attempts", 0)}
        await asyncio.sleep(0)
    yield {
        "type": "done",
        "status": final.get("status"),
        "attempts": final.get("attempts"),
        "code": final.get("code", ""),
        "explanation": final.get("explanation", ""),
        "output": final.get("execution_result", ""),
        "error": final.get("error", ""),
        "review_feedback": final.get("review_feedback", ""),
    }
