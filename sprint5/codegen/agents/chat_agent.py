"""Notebook 1 — Code generator with tools.

bind_tools + ToolNode agent loop over MessagesState, streamed with astream_events,
multi-turn memory via a checkpointer (one thread per conversation).
"""

from __future__ import annotations

from typing import AsyncGenerator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from . import codebase
from .common import execute_shell as _shell
from .common import get_llm, list_files, load_rules, read_text, rel, write_text

SYSTEM_PROMPT = """You are an expert software engineer working inside a coding workspace, like Claude Code.
Your working directory is the workspace root; always use paths relative to it (e.g. 'my_project/main.py').
Look at what already exists before deciding on a structure, and follow the project's existing conventions.

When generating code:
- Use type hints on all functions
- Add concise docstrings
- Follow PEP 8 conventions
- Prefer modern Python (3.10+) features where appropriate

Use your tools: explore with list_directory / search_codebase, read files before modifying them,
and always write generated code to files with write_file (complete file contents).
Keep your final answer short: summarize what you did."""


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file and return it as a string."""
    return read_text(filepath)


@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file. Creates the file (and parent folders) if it doesn't exist."""
    path = write_text(filepath, content)
    return f"File written: {rel(path)} ({content.count(chr(10)) + 1} lines)"


@tool
def list_directory(directory: str = ".") -> str:
    """List all files (recursively) under the given directory."""
    files = list_files(directory)
    return "\n".join(files) if files else "(empty)"


@tool
def search_codebase(query: str) -> str:
    """Semantic search over the workspace code. Returns the most relevant snippets."""
    docs, _ = codebase.search(query, ".")
    return codebase.format_results(docs) or "No matches."


@tool
def execute_shell(command: str) -> str:
    """Execute a shell command in the workspace and return its output."""
    return _shell(command)


BASE_TOOLS = [read_file, write_file, list_directory, search_codebase]
ALL_TOOLS = BASE_TOOLS + [execute_shell]


async def agent(state: MessagesState, config: RunnableConfig):
    c = config.get("configurable", {})
    tools = ALL_TOOLS if c.get("allow_shell") else BASE_TOOLS
    system = SYSTEM_PROMPT
    rules = load_rules(c.get("rules", ""))
    if rules:
        system += f"\n\nProject rules (always follow):\n{rules}"
    llm = get_llm(c.get("model")).bind_tools(tools)
    response = await llm.ainvoke([SystemMessage(content=system)] + state["messages"], config)
    return {"messages": [response]}


def should_continue(state: MessagesState):
    return "tools" if state["messages"][-1].tool_calls else END


_g = StateGraph(MessagesState)
_g.add_node("agent", agent)
_g.add_node("tools", ToolNode(ALL_TOOLS))
_g.add_edge(START, "agent")
_g.add_conditional_edges("agent", should_continue, ["tools", END])
_g.add_edge("tools", "agent")
graph = _g.compile(checkpointer=MemorySaver())


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


async def stream_chat(
    message: str, thread_id: str, model: str | None, rules: str, allow_shell: bool
) -> AsyncGenerator[dict, None]:
    config = {
        "configurable": {"thread_id": thread_id, "model": model, "rules": rules, "allow_shell": allow_shell},
        "recursion_limit": 50,
    }
    inputs = {"messages": [HumanMessage(content=message)]}
    async for ev in graph.astream_events(inputs, config, version="v2"):
        kind = ev["event"]
        if kind == "on_chat_model_stream":
            text = _text(ev["data"]["chunk"].content)
            if text:
                yield {"type": "token", "text": text}
        elif kind == "on_chat_model_end":
            yield {"type": "message_end"}
        elif kind == "on_tool_start":
            yield {"type": "tool_start", "id": ev["run_id"], "name": ev["name"], "input": ev["data"].get("input", {})}
        elif kind == "on_tool_end":
            out = ev["data"].get("output")
            out = getattr(out, "content", out)
            yield {"type": "tool_end", "id": ev["run_id"], "name": ev["name"], "output": str(out)[:4000]}
    yield {"type": "done"}
