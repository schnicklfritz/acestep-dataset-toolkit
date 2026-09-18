#!/usr/bin/env python
"""Print a Kaggle kernel's status and log tail.

Use this when a run failed and the app's error did not include the log (e.g. the
run happened before the diagnostics landed, or the app was restarted and lost
the kernel reference).

    .venv/bin/python scripts/moss_kernel_log.py <owner>/<kernel-slug>

Find the slug in the Kaggle URL: kaggle.com/code/<owner>/<kernel-slug>
MOSS runs are named ``ace-moss-XXXXXX``.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import DEFAULT_CONFIG                      # noqa: E402
from modules.config_store import load_config           # noqa: E402
from modules.kaggle import (                            # noqa: E402
    fetch_kernel_logs,
    kernel_status_text,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="owner/kernel-slug, e.g. you/ace-moss-a1b2c3")
    parser.add_argument("--chars", type=int, default=12000,
                        help="how many trailing log characters to show")
    parser.add_argument("--full", action="store_true",
                        help="show the whole log, not just the tail")
    args = parser.parse_args(argv)

    if "/" not in args.kernel:
        print("Kernel must be owner/slug (see kaggle.com/code/<owner>/<slug>)")
        return 2

    config = load_config(DEFAULT_CONFIG)
    if not config.get("kaggle_user") or not config.get("kaggle_key"):
        print("No Kaggle credentials in Settings -- cannot read the log.")
        return 2

    owner, slug = args.kernel.split("/", 1)

    status = kernel_status_text(config, slug)
    print(f"kernel : {args.kernel}")
    print(f"status : {status or '(unavailable)'}")
    print()

    text = fetch_kernel_logs(
        config, slug, max_chars=(10 ** 9 if args.full else args.chars)
    )
    if not text:
        print("No log returned. Check the slug, and that the kernel is yours.")
        return 1

    if not args.full:
        print(f"--- last {args.chars} characters of the log ---")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
