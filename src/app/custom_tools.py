"""Tasks 7 and 8: kb_stats, document_outline, delete_document, update_document (all async, all per user)."""
import asyncio
import time

from langchain.tools import ToolRuntime
from langchain_core.tools import ToolException, tool
from opentelemetry import metrics

from app.context import Ctx
from app.errors import to_tool_exception
from app.kb import files
from app.kb.pipeline import ingest_file
from app.kb.store import KnowledgeBase
from app.telemetry import log

_meter = metrics.get_meter("agent")
KBSTATS_CALLS = _meter.create_counter("kbstats.calls", unit="1", description="kb_stats calls")
OUTLINE_CALLS = _meter.create_counter("outline.calls", unit="1", description="document_outline calls")


def _unknown_document(doc_name: str, docs: list[dict]) -> ToolException:
    names = ", ".join(f'"{d["filename"]}"' for d in docs) or "(none)"
    return ToolException(f"ERROR[UNKNOWN_DOCUMENT]: No document matches '{doc_name}' in your knowledge base. Documents: {names}. "
                         "Next step: retry with one of these names.")


async def _find(kb: KnowledgeBase, user_id: str, doc_name: str) -> tuple[str, dict]:
    """doc_name -> (exact file name, document row) in THIS user's knowledge base, or ERROR[UNKNOWN_DOCUMENT]."""
    exact = await asyncio.to_thread(kb.resolve_doc_name, user_id, doc_name)
    docs = await asyncio.to_thread(kb.documents, user_id)
    doc = next((d for d in docs if d["filename"] == exact), None) if exact else None
    if doc is None:
        raise _unknown_document(doc_name, docs)
    return exact, doc


def build_custom_tools(kb: KnowledgeBase) -> list:
    @tool
    async def kb_stats(runtime: ToolRuntime[Ctx], top_keywords: int = 5) -> str:
        """Overall numbers about the user's knowledge base: how many documents, sections and chunks it holds, and its most
        common keywords. Use it when the user asks how big the knowledge base is, what it covers overall or what the main topics are.

        Args:
            top_keywords: how many top keywords to list (1 to 20, default 5)
        """
        t0, status = time.perf_counter(), "error"
        try:
            if not 1 <= top_keywords <= 20:
                raise ToolException("ERROR[INVALID_INPUT]: top_keywords must be between 1 and 20. Next step: retry with a number from 1 to 20.")
            stats = await asyncio.to_thread(kb.stats, runtime.context.user_id)
            status = "ok"
            if not stats["documents"]:
                return "The knowledge base is empty."
            return (f"{stats['documents']} documents, {stats['sections']} sections, {stats['chunks']} chunks. "
                    f"Top keywords: {', '.join(stats['keywords'][:top_keywords])}.")
        finally:
            KBSTATS_CALLS.add(1, {"status": status})
            log.info("kb_stats_done", tool="kb_stats", status=status, duration_ms=int((time.perf_counter() - t0) * 1000))

    @tool
    async def document_outline(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """The structure of ONE document in the user's knowledge base: its sections in reading order with level, pages and a
        summary. Use it when the user asks what a document contains, how it is organised, or which section covers something
        (then search that section).

        Args:
            doc_name: file name from the KNOWLEDGE BASE list
        """
        t0, status, n_sections = time.perf_counter(), "error", 0
        try:
            user_id = runtime.context.user_id
            exact, doc = await _find(kb, user_id, doc_name)
            sections = await asyncio.to_thread(kb.outline, user_id, doc["doc_id"])
            lines = [f"Outline of '{exact}' ({len(sections)} sections):"]
            for i, s in enumerate(sections, start=1):
                if s["page_start"] is None:
                    pages = ""
                elif s["page_start"] != s["page_end"]:
                    pages = f" (p.{s['page_start']}-{s['page_end']})"
                else:
                    pages = f" (p.{s['page_start']})"
                lines.append(f"{i}. [L{s['level']}] {s['heading']}{pages}: {s['summary']}")
            n_sections, status = len(sections), "ok"
            return "\n".join(lines)
        finally:
            OUTLINE_CALLS.add(1, {"status": status})
            log.info("document_outline_done", tool="document_outline", status=status,
                     duration_ms=int((time.perf_counter() - t0) * 1000), sections=n_sections)

    @tool
    async def delete_document(doc_name: str, runtime: ToolRuntime[Ctx]) -> str:
        """Remove ONE document from the user's knowledge base, so it is no longer searched or listed. Use it when the user asks
        to delete, remove or forget a document.

        Args:
            doc_name: file name from the KNOWLEDGE BASE list
        """
        user_id = runtime.context.user_id
        exact, doc = await _find(kb, user_id, doc_name)
        try:
            await asyncio.to_thread(kb.delete_document, user_id, doc["doc_id"])
            left = len(await asyncio.to_thread(kb.documents, user_id))
        except Exception as exc:
            raise to_tool_exception(exc) from exc
        return f"Deleted '{exact}'. {left} documents left."

    @tool
    async def update_document(doc_name: str, source: str, runtime: ToolRuntime[Ctx]) -> str:
        """Replace a document in the user's knowledge base with a newly uploaded version under the same name. Use it when the
        user uploads a new version of a document that is already in the knowledge base.

        Args:
            doc_name: file name of the existing document, from the KNOWLEDGE BASE list
            source: the file_id of the NEW upload (from the upload message)
        """
        user_id = runtime.context.user_id
        exact, old = await _find(kb, user_id, doc_name)
        try:
            path, _ = files.resolve(runtime.context.session_id, source.strip())
            result, existing = await ingest_file(kb, user_id, path, exact, runtime.config)  # new version first ...
            if existing:
                return f"'{exact}' is already up to date."
            await asyncio.to_thread(kb.delete_document, user_id, old["doc_id"])  # ... the old one only after it worked
        except Exception as exc:
            raise to_tool_exception(exc) from exc
        return f"Updated '{exact}': {len(result.sections)} sections, {result.n_chunks} chunks."

    tools = [kb_stats, document_outline, delete_document, update_document]
    for t in tools:
        t.handle_tool_error = True
    return tools
