"""ACE-Step Dataset Toolkit — MCP server.

Exposes the app's dataset tools over the **Model Context Protocol** (stdio) so
any MCP client (Claude Desktop, Cursor, custom agents) can inspect, audit, tag
and curate your datasets directly.

Requires:  pip install "mcp[cli]"
Run:       python mcp_server.py --dataset path/to/dataset.json
"""
import argparse
import sys

DEFAULT_DATASET = "dataset.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET,
        help="path to the dataset JSON to expose (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print("MCP server needs:  pip install 'mcp[cli]'", file=sys.stderr)
        sys.exit(1)

    from modules import mcp_tools

    mcp = FastMCP("ACE-Step Dataset Toolkit")

    @mcp.tool()
    def list_tracks() -> str:
        """List the tracks in the dataset with their captions."""
        return mcp_tools.tool_list_tracks(args.dataset)

    @mcp.tool()
    def dataset_summary() -> str:
        """High-level dataset summary: track count, vocal/instrumental mix, caption coverage."""
        return mcp_tools.tool_dataset_summary(args.dataset)

    @mcp.tool()
    def health_audit() -> str:
        """Audit the dataset: missing files, mono/short tracks, undetermined BPM, near-duplicates."""
        return mcp_tools.tool_health_audit(args.dataset)

    @mcp.tool()
    def tag_track(audio_path: str) -> str:
        """Tag an audio file: BPM, key, and detected instruments."""
        return mcp_tools.tool_tag_track(audio_path)

    @mcp.tool()
    def curate_dataset(target_sound: str) -> str:
        """Recommend what to add so the dataset converges on a target sound."""
        return mcp_tools.tool_curate(args.dataset, target_sound)

    mcp.run()


if __name__ == "__main__":
    main()