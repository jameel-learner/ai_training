"""Shared config, LLM factory, workspace sandbox and code/shell execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

BASE_DIR = Path(__file__).resolve().parent.parent  # .../codegen
load_dotenv(BASE_DIR.parent / ".env")  # the .env one level above codegen/

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = os.getenv("CODEGEN_MODEL", "openai/gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("CODEGEN_EMBEDDING_MODEL", "openai/text-embedding-3-small")

WORKSPACE = (BASE_DIR / "workspace").resolve()
SCRATCH = WORKSPACE / ".scratch"
TEMPLATE_DIR = BASE_DIR / "sample_project_template"
SAMPLE_PROJECT = "sample_project"

SKIP_DIRS = {".git", "__pycache__", ".venv", "venv", "node_modules", ".scratch"}


RULES_FILE = BASE_DIR / "rules.md"


def load_rules(extra: str = "") -> str:
    """Global rules from codegen/rules.md (re-read every call) + any rules set in the UI (/rules)."""
    parts = []
    if RULES_FILE.is_file():
        text = "\n".join(
            l for l in RULES_FILE.read_text(encoding="utf-8").splitlines() if not l.lstrip().startswith("# ")
            and l.strip() != "#"
        ).strip()
        if text:
            parts.append(text)
    if extra and extra.strip():
        parts.append("## Session rules\n" + extra.strip())
    return "\n\n".join(parts)


def get_llm(model: str | None = None) -> ChatOpenAI:
    """ChatOpenAI pointed at OpenRouter (same setup as the notebooks)."""
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            f"OPENROUTER_API_KEY not found. Expected it in {BASE_DIR.parent / '.env'}"
        )
    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        api_key=OPENROUTER_API_KEY,
        base_url=OPENROUTER_BASE_URL,
        temperature=0.2,
        default_headers={"X-Title": "Codegen Studio"},
    )


def structured(model: str | None, schema):
    """with_structured_output via function calling — works across OpenRouter providers."""
    return get_llm(model).with_structured_output(schema, method="function_calling")


# ---------------------------------------------------------------- workspace sandbox
def ensure_workspace() -> None:
    WORKSPACE.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    # Seed the demo project only on first run (empty workspace) — never re-create it behind the user's back.
    if not any(p.is_dir() and p.name not in SKIP_DIRS for p in WORKSPACE.iterdir()):
        reset_sample_project()


def reset_sample_project() -> None:
    target = WORKSPACE / SAMPLE_PROJECT
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(TEMPLATE_DIR, target, ignore=shutil.ignore_patterns("__pycache__"))


def resolve(path: str) -> Path:
    """Resolve a path relative to the workspace and refuse anything outside it."""
    p = Path(path.strip().strip("'\""))
    parts = p.parts
    if parts and parts[0] == "workspace":  # models sometimes prefix the root name
        p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    full = (p if p.is_absolute() else WORKSPACE / p).resolve()
    if full != WORKSPACE and WORKSPACE not in full.parents:
        raise ValueError(f"Path '{path}' is outside the workspace sandbox")
    return full


def rel(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE).as_posix()


def read_text(path: str) -> str:
    return resolve(path).read_text(encoding="utf-8")


def write_text(path: str, content: str) -> Path:
    full = resolve(path)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content, encoding="utf-8", newline="")  # keep \n as-is (no CRLF rewrite on Windows)
    return full


def list_files(root: str = ".") -> list[str]:
    base = resolve(root)
    out: list[str] = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
        for f in sorted(filenames):
            out.append(rel(Path(dirpath) / f))
    return out


# ---------------------------------------------------------------- execution
def _env() -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env.pop("OPENROUTER_API_KEY", None)  # don't hand the key to generated code
    return env


def execute_python(code: str, timeout: int = 10) -> dict:
    """NB2's executor: run code in a subprocess, capture stdout/stderr."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", dir=SCRATCH, delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        script = f.name
    try:
        r = subprocess.run(
            [sys.executable, script],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=SCRATCH,
            env=_env(),
        )
        return {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode}
    except subprocess.TimeoutExpired:
        return {"stdout": "", "stderr": f"TimeoutError: execution exceeded {timeout}s", "returncode": 1}
    finally:
        Path(script).unlink(missing_ok=True)


def execute_shell(command: str, timeout: int = 30, cwd: Path | None = None) -> str:
    """NB3's shell tool, run inside the workspace."""
    try:
        r = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            cwd=cwd or WORKSPACE,
            env=_env(),
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {timeout}s"
    out = r.stdout
    if r.stderr:
        out += f"\nSTDERR: {r.stderr}"
    out += f"\n(exit code {r.returncode})"
    return out.strip()
