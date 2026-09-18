"""Tests for the MCP servers.

These call tools through the real MCP ``call_tool`` boundary rather than the
underlying functions, so a wiring mistake (a tool that silently returns the
wrong thing, or a default that ignores the CLI config) fails here.
"""
import asyncio
import re

import pytest

import mcp_research_server
from modules.mcp_compat import load_server_class


def _call(server, name, **arguments):
    """Invoke a registered MCP tool synchronously and return its text."""
    result = asyncio.run(server.call_tool(name, arguments))
    return result.content[0].text


def _tool_names(server):
    return sorted(t.name for t in asyncio.run(server.list_tools()))


@pytest.fixture
def research_server(tmp_path):
    """A research server pointed at a small corpus + vocabulary."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.txt").write_text(
        "doom metal review: thunderous drums and downtuned guitar.\n"
        "Raw 1970s analog production.\n",
        encoding="utf-8",
    )
    (corpus / "b.txt").write_text(
        "another doom metal piece with thunderous drums.\n",
        encoding="utf-8",
    )
    vocab = tmp_path / "vocabulary.txt"
    vocab.write_text("thunderous drums\ndowntuned guitar\ngated reverb\n",
                     encoding="utf-8")
    return mcp_research_server.build_server(str(corpus), str(vocab)), str(corpus)


# --------------------------------------------------------------------------
# compat shim (mcp 1.x FastMCP / mcp 2.x MCPServer)
# --------------------------------------------------------------------------

def test_load_server_class_is_usable():
    cls = load_server_class()
    assert hasattr(cls, "tool")
    assert hasattr(cls, "run")
    assert hasattr(cls, "call_tool")


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

EXPECTED_TOOLS = {
    "corpus_stats",
    "grep_corpus",
    "induct_terms",
    "unlisted_terms",
    "vocab_diff",
    "spec_coverage",
    "lexicon_report",
    "kaggle_kernel_status",
    "kaggle_kernel_log",
}


def test_research_server_registers_every_tool(research_server):
    server, _corpus = research_server
    assert set(_tool_names(server)) == EXPECTED_TOOLS


def test_default_corpus_is_used_when_argument_empty(research_server):
    server, _corpus = research_server
    out = _call(server, "corpus_stats")          # no corpus_dir argument
    assert "Files: 2" in out


def test_explicit_corpus_argument_overrides_default(research_server, tmp_path):
    server, _corpus = research_server
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.txt").write_text("hello\n", encoding="utf-8")
    out = _call(server, "corpus_stats", corpus_dir=str(other))
    assert "Files: 1" in out


def test_induct_terms_through_mcp(research_server):
    server, _corpus = research_server
    out = _call(server, "induct_terms")
    assert "thunderous drums" in out
    assert "gated reverb" not in out             # absent from the corpus


def test_unlisted_terms_through_mcp(research_server):
    server, _corpus = research_server
    out = _call(server, "unlisted_terms")
    assert "doom metal" in out


def test_grep_corpus_through_mcp_is_capped(research_server):
    server, _corpus = research_server
    out = _call(server, "grep_corpus", pattern="doom", max_hits=1)
    hits = [ln for ln in out.splitlines() if re.match(r"^\S+:\d+:", ln)]
    assert len(hits) == 1
    assert "capped at 1" in out


def test_token_discipline_holds_across_the_mcp_boundary(research_server):
    """The MCP surface must not offer a way to bulk-read the corpus."""
    server, _corpus = research_server
    for name in ("corpus_stats", "induct_terms", "unlisted_terms"):
        out = _call(server, name)
        assert "Raw 1970s analog production" not in out, f"{name} leaked prose"


def test_missing_corpus_returns_error_not_exception():
    server = mcp_research_server.build_server("/nonexistent/corpus/xyz")
    out = _call(server, "corpus_stats")
    assert out.startswith("ERROR:")


def test_lexicon_report_tool(tmp_path):
    doc = tmp_path / "doc.md"
    doc.write_text("### Vocal\nwhispered, belted, falsetto\n", encoding="utf-8")
    server = mcp_research_server.build_server(str(tmp_path))
    out = _call(server, "lexicon_report", source_path=str(doc),
                out_dir=str(tmp_path / "out"))
    assert "Descriptors: 3" in out


def test_lexicon_report_missing_doc_reports_error(tmp_path):
    server = mcp_research_server.build_server(str(tmp_path))
    out = _call(server, "lexicon_report", source_path=str(tmp_path / "absent.md"))
    assert out.startswith("ERROR:")


# --------------------------------------------------------------------------
# Kaggle tools -- must degrade cleanly without credentials, never touch network
# --------------------------------------------------------------------------

def test_kaggle_tools_report_missing_credentials(monkeypatch, tmp_path):
    monkeypatch.setattr("modules.config_store.load_config", lambda *a, **k: {})
    server = mcp_research_server.build_server(str(tmp_path))
    assert "No Kaggle credentials" in _call(server, "kaggle_kernel_status",
                                            kernel="me/ace-moss-abc123")
    assert "No Kaggle credentials" in _call(server, "kaggle_kernel_log",
                                            kernel="me/ace-moss-abc123")


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
