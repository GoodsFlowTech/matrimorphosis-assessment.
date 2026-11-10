"""Task 9: the notes and tasks MCP server. Run by docker compose / CI as:  python src/app/mcp_server/server.py --http  (port 9000, path /mcp)

The storage is PROVIDED (taskstore.py: TaskStore), the user lookup is PROVIDED (app/users.py: UserStore.user_for_token).
You write the integration: register the eight tools below on `mcp`, and make every call run AS THE CALLING USER.

How the server knows the user (task 9)
  The agent forwards the logged-in user's token on every MCP call as the HTTP header  Authorization: Bearer <token>  (you write that
  half in agent.py, task 9). A tool declares a `ctx: Context` parameter; FastMCP injects it and the HTTP request is
  ctx.request_context.request (Starlette: .headers). Write ONE helper `current_user_id(ctx) -> str` that
    - reads the bearer token, looks it up with UserStore().user_for_token(token)
    - raises ToolError("ERROR[UNAUTHENTICATED]: <what>. Next step: log in and retry.") when the header is missing, malformed or unknown
  and call it first in every tool. Never take a user id from a tool argument: the model must not be able to name another user.

The eight tools (names and arguments are fixed; each returns json.dumps(<what the TaskStore method returned>), a string)
  add_note(title: str, body: str)               list_notes()                          delete_note(note_id: int)
  add_task(title: str, details: str = "")       list_tasks(status: str = "")          set_task_status(task_id: int, status: str)
  delete_task(task_id: int)                     task_summary()
  status is "todo" | "in_progress" | "done". TaskStore raises TaskError for bad input: re-raise it as ToolError(str(error)).
  The docstring of each tool is its prompt for the model: say WHEN to use it and describe the arguments.
"""
import json  # noqa: F401  (you will need it)
import os
import sys

from mcp.server.fastmcp import Context, FastMCP  # noqa: F401
from mcp.server.fastmcp.exceptions import ToolError  # noqa: F401

from app.mcp_server.taskstore import TaskError, TaskStore  # noqa: F401
from app.users import UserStore  # noqa: F401

mcp = FastMCP("tasks", host="0.0.0.0", port=int(os.getenv("MCP_PORT", "9000")))
store = TaskStore()

def current_user_id(ctx: Context) -> str:
    """The id of the calling user, from `Authorization: Bearer <token>`. Never from a tool argument."""
    request = getattr(ctx.request_context, "request", None)
    header = request.headers.get("authorization", "") if request is not None else ""
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token:
        raise ToolError("ERROR[UNAUTHENTICATED]: the request carries no valid bearer token. Next step: log in and retry.")
    user = UserStore().user_for_token(token)
    if user is None:
        raise ToolError("ERROR[UNAUTHENTICATED]: the token is unknown or has been revoked. Next step: log in and retry.")
    return user["user_id"]


def _call(method, *args) -> str:
    try:
        return json.dumps(method(*args))
    except TaskError as error:
        raise ToolError(str(error)) from error


@mcp.tool()
def add_note(title: str, body: str, ctx: Context) -> str:
    """Save a note for the user. Use it when the user wants something written down or remembered.

    Args:
        title: short title of the note
        body: the text of the note
    """
    return _call(store.add_note, current_user_id(ctx), title, body)


@mcp.tool()
def list_notes(ctx: Context) -> str:
    """List all of the user's notes (id, title, body). Use it to show or find notes, or to get a note id."""
    return _call(store.list_notes, current_user_id(ctx))


@mcp.tool()
def delete_note(note_id: int, ctx: Context) -> str:
    """Delete one of the user's notes.

    Args:
        note_id: the id of the note (see list_notes)
    """
    return _call(store.delete_note, current_user_id(ctx), note_id)


@mcp.tool()
def add_task(title: str, ctx: Context, details: str = "") -> str:
    """Create a task for the user, starting in status todo. Use it when the user wants to add something to their to-do list.

    Args:
        title: short title of the task
        details: optional longer description
    """
    return _call(store.add_task, current_user_id(ctx), title, details)


@mcp.tool()
def list_tasks(ctx: Context, status: str = "") -> str:
    """List the user's tasks (id, title, details, status). Use it to show tasks or to get a task id.

    Args:
        status: optional filter: "todo", "in_progress" or "done"; empty lists all tasks
    """
    return _call(store.list_tasks, current_user_id(ctx), status)


@mcp.tool()
def set_task_status(task_id: int, status: str, ctx: Context) -> str:
    """Move a task to a new status. Tasks move one step at a time: todo -> in_progress -> done (todo cannot jump to done).

    Args:
        task_id: the id of the task (see list_tasks)
        status: "todo", "in_progress" or "done"
    """
    return _call(store.set_task_status, current_user_id(ctx), task_id, status)


@mcp.tool()
def delete_task(task_id: int, ctx: Context) -> str:
    """Delete one of the user's tasks.

    Args:
        task_id: the id of the task (see list_tasks)
    """
    return _call(store.delete_task, current_user_id(ctx), task_id)


@mcp.tool()
def task_summary(ctx: Context) -> str:
    """How many of the user's tasks are in each status (todo, in_progress, done). Use it for an overview of progress."""
    return _call(store.summary, current_user_id(ctx))


if __name__ == "__main__":
    # stdio cannot carry the Authorization header: the agent talks to this server over HTTP (MCP_URL)
    mcp.run(transport="streamable-http" if "--http" in sys.argv else "stdio")
