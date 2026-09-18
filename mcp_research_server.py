"""ACE-Step Dataset Toolkit — corpus research MCP server.

Exposes the corpus-research tools over the **Model Context Protocol** (stdio)
so any MCP client (Claude Desktop, Cursor, Cline, custom agents) can interrogate
a directory of source documents — album reviews, session notes, interviews, fan
descriptions — and induct the vocabulary those sources actually use.

Separate from ``mcp_server.py`` because it is a different concern (corpus
research vs dataset inspection) with a different lifecycle (one-off vocabulary
induction vs per-dataset access).

    pip install "mcp[cli]"
    python mcp_research_server.py --corpus /path/to/research/black_sabbath

Every tool is capped and NONE returns raw document text, so connecting an agent
to a multi-megabyte corpus cannot blow its context window. See
``modules/research_tools.py`` for the guarantees and their rationale.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import research_tools                              # noqa: E402
from modules.mcp_compat import load_server_class                # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CORPUS = os.path.join(ROOT, "docs", "research")
DEFAULT_VOCAB = os.path.join(ROOT, "docs", "vocabulary.txt")


def build_server(default_corpus=DEFAULT_CORPUS, default_vocab=DEFAULT_VOCAB):
    """Construct the MCP server with every research tool registered.

    Split out from ``main`` so tool wiring can be exercised in tests without
    starting a stdio transport.
    """
    ServerClass = load_server_class()
    mcp = ServerClass("ACE-Step Corpus Research")

    def _corpus(corpus_dir):
        """An empty argument means "use the server's configured corpus"."""
        return corpus_dir or default_corpus

    def _kaggle_config():
        """Return a config dict, or an error string if credentials are absent."""
        from config import DEFAULT_CONFIG
        from modules.config_store import load_config
        config = load_config(DEFAULT_CONFIG)
        if not config.get("kaggle_user") or not config.get("kaggle_key"):
            return "No Kaggle credentials in Settings -- cannot reach Kaggle."
        return config

    @mcp.tool()
    def corpus_stats(corpus_dir: str = "") -> str:
        """Counts only: files, characters, extensions. Returns no document content."""
        return research_tools.corpus_stats(_corpus(corpus_dir))

    @mcp.tool()
    def grep_corpus(pattern: str, corpus_dir: str = "", max_hits: int = 50) -> str:
        """Return only the lines matching a regex, as `file:line: text`, capped.

        The one sanctioned way to read corpus prose: you must state the pattern,
        and you only get back lines that matched it.
        """
        return research_tools.grep_corpus(_corpus(corpus_dir), pattern, max_hits)

    @mcp.tool()
    def induct_terms(corpus_dir: str = "", vocab_path: str = "",
                     min_sources: int = 2, top: int = 40) -> str:
        """Rank known vocabulary terms by how many corpus sources actually use them.

        The core tool: shows which descriptors are grounded in real descriptions
        of this music, and which are aspirational.
        """
        return research_tools.induct_terms(
            _corpus(corpus_dir), vocab_path or default_vocab, min_sources, top
        )

    @mcp.tool()
    def unlisted_terms(corpus_dir: str = "", vocab_path: str = "", n: int = 30) -> str:
        """Frequent 1-3 word phrases the corpus uses that the vocabulary LACKS.

        Community terminology your curated list is missing -- how the vocabulary
        grows from evidence rather than guesswork.
        """
        return research_tools.unlisted_terms(
            _corpus(corpus_dir), vocab_path or default_vocab, n
        )

    @mcp.tool()
    def vocab_diff(a_path: str, b_path: str) -> str:
        """Compare two vocabulary files: shared terms and what each has alone."""
        return research_tools.vocab_diff(a_path, b_path)

    @mcp.tool()
    def spec_coverage(spec_path: str, vocab_path: str = "") -> str:
        """Which vocabulary terms a caption spec uses, and which it never uses."""
        return research_tools.spec_coverage(spec_path, vocab_path or default_vocab)

    @mcp.tool()
    def lexicon_report(source_path: str = "", out_dir: str = "") -> str:
        """Rebuild the vocabulary + section-marker files from the descriptor doc."""
        try:
            result = research_tools.build_lexicon(
                source_path or os.path.join(ROOT, "docs", "descriptor_reference.md"),
                out_dir or None,
            )
        except FileNotFoundError as exc:
            return f"ERROR: {exc}"
        return result["report"]

    @mcp.tool()
    def kaggle_kernel_status(kernel: str) -> str:
        """Status of a Kaggle kernel, e.g. 'owner/ace-moss-a1b2c3'. Needs Kaggle keys."""
        config = _kaggle_config()
        if isinstance(config, str):
            return config
        slug = kernel.split("/", 1)[1] if "/" in kernel else kernel
        from modules.kaggle import kernel_status_text
        return kernel_status_text(config, slug) or "(status unavailable)"

    @mcp.tool()
    def kaggle_kernel_log(kernel: str, chars: int = 4000) -> str:
        """Tail of a Kaggle kernel's log, for diagnosing a failed run."""
        config = _kaggle_config()
        if isinstance(config, str):
            return config
        slug = kernel.split("/", 1)[1] if "/" in kernel else kernel
        from modules.kaggle import fetch_kernel_logs
        text = fetch_kernel_logs(config, slug, max_chars=max(1, chars))
        return text or "(no log returned -- check the slug and that the kernel is yours)"

    return mcp


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=DEFAULT_CORPUS,
                        help="default corpus directory (default: %(default)s)")
    parser.add_argument("--vocab", default=DEFAULT_VOCAB,
                        help="default vocabulary file (default: %(default)s)")
    args = parser.parse_args(argv)

    try:
        mcp = build_server(args.corpus, args.vocab)
    except ImportError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    mcp.run()


if __name__ == "__main__":
    main()
