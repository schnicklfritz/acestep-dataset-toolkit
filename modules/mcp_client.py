"""MCP client — lets the Assistant use any Model Context Protocol server.

WHY THIS EXISTS
---------------
The app already *ships* two MCP servers (``mcp_server.py`` for dataset tools,
``mcp_research_server.py`` for corpus/vocabulary research), but they were only
reachable from another program — Claude Desktop, Cursor, Cline. This module
makes the relationship bidirectional: the Assistant in the app can be *given*
those tools, plus any other MCP server you add to settings, so the model can
reach beyond what the app implements natively (filesystem, web search, DSP, a
research corpus…).

DESIGN NOTES
------------
  * **Namespacing is mandatory.** Two servers both plausibly export
    ``list_tracks``; OpenAI-compatible tool names must be unique inside one
    call, so every spec is exposed as ``<server>__<tool>`` and routed back
    through :func:`split_tool_name`. A collision would silently shadow one tool
    with another's — the model would call the wrong tool and be told it worked.
  * **Spec conversion is pure.** MCP hands over JSON-Schema parameter objects;
    the OpenAI function format wants a JSON Schema under ``parameters``. Being
    a pure function makes the translation testable without a running server.
  * **The ``mcp`` package is optional.** Import it lazily so the app still runs
    without it, and say exactly what to install when it is missing rather than
    raising a bare ImportError from inside a worker thread.
"""
from __future__ import annotations

import json

# Config key: a plain list, so it is hand-editable in settings.json.
MCP_SERVERS_KEY = "mcp_servers"

# Documented example, so a user can see the shape without reading this file:
#   {"name": "dataset", "command": "python", "args": ["mcp_server.py",
#    "--dataset", "dataset.json"]}
DEFAULT_SERVERS = []


class McpUnavailable(RuntimeError):
    """The optional ``mcp[cli]`` package is not installed."""


def _require_mcp():
    try:
        from mcp import ClientSession, StdioServerParameters  # noqa: F401
        from mcp.client.stdio import stdio_client  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise McpUnavailable(
            "The 'mcp' package is not installed. Run: "
            'pip install "mcp[cli]" — then add servers under the MCP section '
            "in ⚙ Settings."
        ) from exc
    return ClientSession, StdioServerParameters, stdio_client


# ---------------------------------------------------------------------------
# Server registry (config in, normalised entries out)
# ---------------------------------------------------------------------------
def parse_mcp_servers(raw):
    """Normalise whatever the config holds into ``[{name, command, args, env}]``.

    Accepts a list of dicts, a JSON string, or ``None``. Invalid entries are
    dropped rather than raising: one bad line in settings.json must not take the
    whole Assistant panel down with it.
    """
    if not raw:
        return []
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    out, seen = [], set()
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            continue
        command = str(entry.get("command") or "").strip()
        if not command:
            continue
        args = entry.get("args") or []
        if not isinstance(args, (list, tuple)):
            continue
        env = entry.get("env") or {}
        if not isinstance(env, dict):
            env = {}
        name = str(entry.get("name") or "").strip() or f"server{i + 1}"
        if name in seen:                      # duplicate names would shadow each
            continue
        seen.add(name)
        out.append({
            "name": name,
            "command": command,
            "args": [str(a) for a in args],
            "env": {str(k): str(v) for k, v in env.items()},
        })
    return out


# ---------------------------------------------------------------------------
# Spec conversion (pure — this is what the model reads)
# ---------------------------------------------------------------------------
def split_tool_name(spec_name):
    """Split ``<server>__<tool>`` back into ``(server, tool)``.

    A name without the separator maps to ``(None, name)`` so app-native tools
    and MCP tools can live in the same tool list.
    """
    if isinstance(spec_name, str) and "__" in spec_name:
        server, _, tool = spec_name.partition("__")
        if server and tool:
            return server, tool
    return None, spec_name


def tool_spec(server_name, tool):
    """Convert one MCP tool descriptor into an OpenAI function spec.

    MCP calls its parameters ``inputSchema``; OpenAI expects a JSON Schema under
    ``parameters``. Both are JSON Schema, but a server that omits it (legal in
    MCP) must still produce a valid spec — an empty object schema, not ``None``,
    which some providers reject outright.
    """
    if isinstance(tool, dict):
        name = tool.get("name") or tool.get("tool_name") or "tool"
        description = tool.get("description") or ""
        schema = tool.get("inputSchema") or tool.get("parameters") or {}
    else:
        name, description, schema = str(tool), "", {}
    if not isinstance(schema, dict):
        schema = {}
    schema = dict(schema)
    schema.setdefault("type", "object")
    return {
        "type": "function",
        "function": {
            "name": f"{server_name}__{name}",
            "description": str(description)[:512],
            "parameters": schema,
        },
    }


def tool_specs(server_name, tools):
    """Convert a server's tool list into OpenAI specs, namespaced by server."""
    return [tool_spec(server_name, t) for t in (tools or [])]


# ---------------------------------------------------------------------------
# Sessions (the part that needs a running server)
# ---------------------------------------------------------------------------
def _stdio_params(server):
    _ClientSession, StdioServerParameters, _stdio_client = _require_mcp()
    params = StdioServerParameters(
        command=server["command"],
        args=server.get("args") or [],
        env=server.get("env") or None,
    )
    return params


async def _with_session(server, body):
    """Run ``body(session)`` over one stdio session to this server."""
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    params = _stdio_params(server)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await body(session)


def list_tools(server, timeout=30.0):
    """Return a server's tool descriptors as a plain list of dicts.

    Raises ``McpUnavailable`` when the optional package is missing, so the
    caller can report an install command instead of a bare ImportError from a
    worker thread.
    """
    import asyncio

    async def body(session):
        result = await session.list_tools()
        out = []
        for t in result.tools:
            out.append({
                "name": getattr(t, "name", ""),
                "description": getattr(t, "description", "") or "",
                "inputSchema": getattr(t, "inputSchema", {}) or {},
            })
        return out

    try:
        return asyncio.run(asyncio.wait_for(_with_session(server, body), timeout))
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"MCP server '{server['name']}' did not answer list_tools in "
            f"{timeout:.0f}s — check the command in ⚙ Settings."
        ) from exc
    except McpUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — one dead server must not kill the panel
        raise RuntimeError(
            f"MCP server '{server['name']}' failed: {exc}"
        ) from exc


def call_tool(server, tool_name, arguments, timeout=60.0):
    """Invoke ``tool_name`` on ``server`` and return its text result.

    MCP results come back as a list of content blocks (text / image / …); only
    text blocks are concatenated, and non-text blocks are named rather than
    dropped, so a server that answers with an image says so instead of
    appearing to return nothing.
    """
    import asyncio

    async def body(session):
        result = await session.call_tool(tool_name, arguments or {})
        parts, other = [], []
        for block in getattr(result, "content", []) or []:
            kind = getattr(block, "type", "text")
            if kind == "text" and getattr(block, "text", None) is not None:
                parts.append(block.text)
            else:
                other.append(kind)
        if not parts:
            return "(no text content)" + (
                f" — server returned: {', '.join(other)}" if other else ""
            )
        return "\n".join(parts)

    try:
        return asyncio.run(asyncio.wait_for(_with_session(server, body), timeout))
    except asyncio.TimeoutError as exc:
        raise RuntimeError(
            f"MCP tool '{tool_name}' timed out after {timeout:.0f}s on "
            f"'{server['name']}'."
        ) from exc


def gather_specs(servers, timeout=30.0):
    """Build OpenAI specs for every configured server.

    A server that will not start is skipped with a note rather than raising:
    adding a broken server to settings must not stop the Assistant from
    working at all. Returns ``(specs, problems)``.
    """
    specs, problems = [], []
    for server in servers or []:
        try:
            tools = list_tools(server, timeout=timeout)
        except McpUnavailable as exc:
            problems.append(str(exc))
            break            # every server needs the same package; stop early
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{server['name']}: {exc}")
            continue
        specs.extend(tool_specs(server["name"], tools))
    return specs, problems

