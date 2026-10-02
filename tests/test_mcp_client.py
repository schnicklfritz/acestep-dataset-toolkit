"""MCP client: namespacing, spec conversion, and the optional-package path.

The tests that matter here are the ones a broken server would otherwise hide:

  * two servers can both export ``list_tracks`` — a collision that is not
    namespaced silently shadows one tool with another's, and the model calls
    the wrong one while being told it worked;
  * a missing ``mcp`` package must produce an install command, not an
    ImportError from inside a worker thread;
  * one dead server must not stop the Assistant from working.
"""
import asyncio
import sys
import types

import pytest

from modules import mcp_client as mc


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def test_parses_a_json_string_from_settings():
    raw = '[{"name": "dataset", "command": "python", "args": ["mcp_server.py"]}]'
    out = mc.parse_mcp_servers(raw)
    assert out == [{
        "name": "dataset",
        "command": "python",
        "args": ["mcp_server.py"],
        "env": {},
    }]


def test_accepts_a_single_dict_and_a_none():
    assert mc.parse_mcp_servers({"command": "x"})[0]["command"] == "x"
    assert mc.parse_mcp_servers(None) == []
    assert mc.parse_mcp_servers("") == []


def test_drops_entries_that_could_not_run():
    """A bad line in settings.json must not take the panel down."""
    raw = [
        {"name": "no_command", "args": []},
        {"command": "python", "args": "not-a-list"},
        {"command": "python", "name": "good"},
        "not a dict",
        42,
    ]
    out = mc.parse_mcp_servers(raw)
    assert [s["name"] for s in out] == ["good"]


def test_duplicate_names_are_dropped_because_they_would_shadow():
    raw = [
        {"command": "python", "name": "a"},
        {"command": "other", "name": "a"},
    ]
    assert len(mc.parse_mcp_servers(raw)) == 1


def test_a_missing_name_gets_a_positional_one():
    out = mc.parse_mcp_servers([{"command": "python"}, {"command": "python"}])
    assert [s["name"] for s in out] == ["server1", "server2"]


def test_malformed_json_string_is_not_an_exception():
    assert mc.parse_mcp_servers("{nope") == []


# ---------------------------------------------------------------------------
# Namespacing
# ---------------------------------------------------------------------------
def test_split_tool_name_round_trips():
    server, tool = mc.split_tool_name("dataset__list_tracks")
    assert (server, tool) == ("dataset", "list_tracks")


def test_a_native_tool_name_has_no_server():
    assert mc.split_tool_name("find_gaps") == (None, "find_gaps")


def test_specs_from_two_servers_never_collide_on_name():
    """Both servers export list_tracks; one spec must not overwrite the other."""
    tools = [{"name": "list_tracks", "description": "x", "inputSchema": {}}]
    specs = mc.tool_specs("dataset", tools) + mc.tool_specs("research", tools)
    names = [s["function"]["name"] for s in specs]
    assert len(names) == len(set(names))
    assert names == ["dataset__list_tracks", "research__list_tracks"]


# ---------------------------------------------------------------------------
# Spec conversion
# ---------------------------------------------------------------------------
def test_spec_wraps_mcp_input_schema_as_openai_parameters():
    spec = mc.tool_spec("dataset", {
        "name": "tag_track",
        "description": "Tag an audio file: BPM, key, instruments.",
        "inputSchema": {"type": "object", "properties": {"audio_path": {"type": "string"}}},
    })
    fn = spec["function"]
    assert spec["type"] == "function"
    assert fn["name"] == "dataset__tag_track"
    assert fn["description"].startswith("Tag an audio file")
    assert fn["parameters"]["properties"]["audio_path"]["type"] == "string"


def test_a_tool_without_a_schema_still_gets_a_valid_object_schema():
    """Some providers reject a missing/null schema outright."""
    spec = mc.tool_spec("s", {"name": "noop"})
    assert spec["function"]["parameters"] == {"type": "object"}


def test_a_bare_string_tool_is_readable():
    spec = mc.tool_spec("s", "just_a_name")
    assert spec["function"]["name"] == "s__just_a_name"
    assert spec["function"]["description"] == ""


def test_descriptions_are_bounded():
    spec = mc.tool_spec("s", {"name": "t", "description": "x" * 9000})
    assert len(spec["function"]["description"]) <= 512


def test_an_input_schema_type_other_than_object_is_preserved_as_is():
    spec = mc.tool_spec("s", {"name": "t", "inputSchema": {"type": "array"}})
    assert spec["function"]["parameters"]["type"] == "array"


# ---------------------------------------------------------------------------
# Session plumbing (the mcp package is faked, not installed)
# ---------------------------------------------------------------------------
class _Tool:
    def __init__(self, name, description="", inputSchema=None):
        self.name = name
        self.description = description
        self.inputSchema = inputSchema


class _Block:
    def __init__(self, type="text", text=None):
        self.type = type
        self.text = text


class _Result:
    def __init__(self, content):
        self.content = content


class _Session:
    def __init__(self, tools=(), result=None):
        self._tools = list(tools)
        self._result = result
        self.called = None

    async def list_tools(self):
        return types.SimpleNamespace(tools=self._tools)

    async def call_tool(self, name, arguments):
        self.called = (name, arguments)
        return self._result


def _fake_session(monkeypatch, session):
    """Route _with_session at a fake, skipping the subprocess entirely."""
    async def fake(server, body, *a, **k):
        return await body(session)

    monkeypatch.setattr(mc, "_with_session", fake)
    return session


def test_list_tools_returns_plain_dicts(monkeypatch):
    _fake_session(monkeypatch, _Session([_Tool("tag_track", "Tag a file.")]))
    out = mc.list_tools({"name": "s", "command": "python", "args": []})
    assert out == [{"name": "tag_track", "description": "Tag a file.",
                    "inputSchema": {}}]


def test_list_tools_without_a_schema_keeps_an_empty_one(monkeypatch):
    _fake_session(monkeypatch, _Session([_Tool("noop", "", None)]))
    assert mc.list_tools({"name": "s", "command": "python", "args": []})[0][
        "inputSchema"] == {}


def test_call_tool_concatenates_text_blocks(monkeypatch):
    session = _Session(result=_Result([_Block("text", "line 1"),
                                       _Block("text", "line 2")]))
    _fake_session(monkeypatch, session)
    out = mc.call_tool({"name": "s", "command": "python"}, "tag_track",
                       {"audio_path": "a.mp3"})
    assert out == "line 1\nline 2"
    # Arguments must reach the server verbatim.
    assert session.called == ("tag_track", {"audio_path": "a.mp3"})


def test_call_tool_names_a_non_text_reply_instead_of_returning_nothing(monkeypatch):
    _fake_session(monkeypatch, _Session(result=_Result([_Block("image", None)])))
    out = mc.call_tool({"name": "s", "command": "python"}, "render", {})
    assert "no text content" in out and "image" in out


def test_an_empty_result_is_reported(monkeypatch):
    _fake_session(monkeypatch, _Session(result=_Result([])))
    assert mc.call_tool({"name": "s", "command": "python"}, "t", {}) == \
        "(no text content)"


def test_a_server_that_wont_start_is_a_problem_not_an_exception(monkeypatch):
    """One broken server must not stop the Assistant from working."""
    def boom(server, **k):
        raise RuntimeError("exit code 1")

    monkeypatch.setattr(mc, "list_tools", boom)
    specs, problems = mc.gather_specs(
        [{"name": "broken", "command": "python", "args": []}]
    )
    assert specs == []
    assert len(problems) == 1
    assert "broken" in problems[0]


def test_a_working_server_contributes_specs(monkeypatch):
    monkeypatch.setattr(
        mc, "list_tools",
        lambda server, **k: [{"name": "frob", "description": "d", "inputSchema": {}}],
    )
    specs, problems = mc.gather_specs(
        [{"name": "srv", "command": "python", "args": []}]
    )
    assert problems == []
    assert specs[0]["function"]["name"] == "srv__frob"


def test_gather_specs_stops_early_when_the_package_is_missing(monkeypatch):
    """Every server needs the same package — do not try them all."""
    calls = []

    def missing(server, **k):
        calls.append(server["name"])
        raise mc.McpUnavailable('pip install "mcp[cli]"')

    monkeypatch.setattr(mc, "list_tools", missing)
    _specs, problems = mc.gather_specs(
        [{"name": "a", "command": "python"},
         {"name": "b", "command": "python"}]
    )
    assert calls == ["a"]
    assert 'pip install' in problems[0]


def test_missing_mcp_package_says_how_to_install(monkeypatch):
    """A bare ImportError from a worker thread is useless to the user."""
    monkeypatch.setitem(sys.modules, "mcp", None)
    with pytest.raises(mc.McpUnavailable) as exc:
        mc._require_mcp()
    assert 'pip install "mcp[cli]"' in str(exc.value)

