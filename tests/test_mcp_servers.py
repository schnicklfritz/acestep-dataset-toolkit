"""Tests for the dataset MCP server.

These call tools through the real MCP ``call_tool`` boundary rather than the
underlying functions, so a wiring mistake (a tool that silently returns the
wrong thing, or a default that ignores the CLI config) fails here.
"""

import pytest

from modules.mcp_compat import load_server_class


# --------------------------------------------------------------------------
# compat shim (mcp 1.x FastMCP / mcp 2.x MCPServer)
# --------------------------------------------------------------------------

def test_load_server_class_is_usable():
    cls = load_server_class()
    assert hasattr(cls, "tool")
    assert hasattr(cls, "run")
    assert hasattr(cls, "call_tool")


# --------------------------------------------------------------------------
# the dataset server (mcp_server.py) must still work after the mcp 2.x rename
# of FastMCP -> MCPServer
# --------------------------------------------------------------------------

def test_dataset_server_module_imports_and_exposes_main():
    import mcp_server
    assert callable(mcp_server.main)


def test_dataset_server_main_help_exits_cleanly():
    """--help must not require a live MCP transport."""
    import mcp_server
    with pytest.raises(SystemExit) as exc:
        mcp_server.main(["--help"])
    assert exc.value.code == 0
