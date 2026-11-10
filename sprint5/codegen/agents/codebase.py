"""NB3 'codebase brain': FAISS semantic search over workspace code.

Embeddings go through OpenRouter. If embeddings are unavailable for any reason the
index falls back to a simple keyword scorer so search keeps working.
"""

from __future__ import annotations

import logging
import re
import threading
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .common import EMBEDDING_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL, list_files, resolve

log = logging.getLogger("codebase")

TEXT_EXT = {".py", ".md", ".txt", ".js", ".ts", ".tsx", ".html", ".css", ".json", ".toml", ".yaml", ".yml", ".cfg", ".ini"}
_cache: dict[str, tuple[tuple, object, list[Document], str]] = {}
_lock = threading.Lock()


def _fingerprint(files: list[str]) -> tuple:
    fp = []
    for f in files:
        st = resolve(f).stat()
        fp.append((f, st.st_mtime_ns, st.st_size))
    return tuple(fp)


def _build(root: str):
    files = [f for f in list_files(root) if Path(f).suffix in TEXT_EXT]
    fp = _fingerprint(files)
    cached = _cache.get(root)
    if cached and cached[0] == fp:
        return cached

    docs = []
    for f in files:
        try:
            text = resolve(f).read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": f, "filename": f}))
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)

    index, method = None, "keyword"
    if chunks:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            emb = OpenAIEmbeddings(
                model=EMBEDDING_MODEL,
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL,
                check_embedding_ctx_length=False,  # send raw text, not tiktoken ids
            )
            index = FAISS.from_documents(chunks, emb)
            method = "faiss"
        except Exception as e:  # noqa: BLE001
            log.warning("Embeddings unavailable (%s) — using keyword search", e)
    entry = (fp, index, chunks, method)
    _cache[root] = entry
    return entry


def _keyword(query: str, chunks: list[Document], k: int) -> list[Document]:
    terms = [t for t in re.findall(r"[a-zA-Z_]\w+", query.lower()) if len(t) > 2]
    scored = []
    for c in chunks:
        text = c.page_content.lower()
        score = sum(text.count(t) for t in terms)
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]] or chunks[:k]


def search(query: str, root: str = ".", k: int = 4) -> tuple[list[Document], str]:
    with _lock:
        _, index, chunks, method = _build(root)
    if not chunks:
        return [], method
    if index is not None:
        try:
            return index.similarity_search(query, k=k), method
        except Exception as e:  # noqa: BLE001
            log.warning("FAISS search failed (%s) — keyword fallback", e)
    return _keyword(query, chunks, k), "keyword"


def format_results(docs: list[Document]) -> str:
    return "\n\n".join(f"--- {d.metadata['filename']} ---\n{d.page_content}" for d in docs)
