"""Chat memory: the LangGraph checkpointer that remembers each conversation (thread_id), stored in SQLite (CHECKPOINT_DB)."""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "checkpoints.sqlite")


@asynccontextmanager
async def open_checkpointer():
    """Opens an AsyncSqliteSaver on CHECKPOINT_DB and keeps it open until the `async with` block ends (API shutdown)."""
    Path(CHECKPOINT_DB).resolve().parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(CHECKPOINT_DB) as saver:
        yield saver
