"""Compatibility shim for the MCP server class across SDK majors.

mcp 1.x exposed ``FastMCP`` (``from mcp.server.fastmcp import FastMCP``).
mcp 2.x renamed it to ``MCPServer`` (``from mcp.server.mcpserver import
MCPServer``) and left a raising stub in the old location.

Both classes share the API this repo uses -- ``Server(name)``, the
``@server.tool()`` decorator and ``server.run()`` (stdio) -- so one import
keeps every server here working across both majors without pinning.
"""
import importlib

# Tried in order. Textual module paths (not ``from ... import``) so a missing
# module raises ImportError cleanly and we can fall through to the next.
_CANDIDATES = (
    ("mcp.server.mcpserver", "MCPServer"),   # mcp >= 2
    ("mcp.server.fastmcp", "FastMCP"),       # mcp 1.x
    ("mcp.server.fastmcp.server", "FastMCP"),  # older 1.x layout
)


def load_server_class():
    """Return the available MCP server class.

    Raises ImportError (with install guidance) if no supported MCP SDK is
    importable, so callers can report one clear message.
    """
    for module_name, attr in _CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        server_class = getattr(module, attr, None)
        if server_class is not None:
            return server_class
    raise ImportError(
        "MCP server needs:  pip install 'mcp[cli]'  "
        "(supported: mcp 1.x FastMCP or mcp 2.x MCPServer)"
    )
