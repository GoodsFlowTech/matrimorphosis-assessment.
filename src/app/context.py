from dataclasses import dataclass


@dataclass
class Ctx:
    """Per-request runtime context: tools and the dynamic prompt read the session, the user and the token from here."""

    session_id: str
    user_id: str = ""  # the knowledge base id: one per user, shared by all of that user's chats
    token: str = ""  # the caller's bearer token; only the MCP interceptor reads it (never a tool argument, prompt or log line)
