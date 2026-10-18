"""Unified Lyrics panel — the one surface every lyrics tool lives on.

The Lyrics page is one panel: the tidy options (``ui.lyrics_tab``), the three
structure-tag rules (``ui.tidy_panel``) and the token-budget line under the
waveform, with two inner tabs and a footer that states, in one line, the rules a
tidy pass will apply.

======================  =========================================
Inner tab               Built by / wired to
======================  =========================================
Normalize & Tidy        ``ui.lyrics_tab.build_lyrics_tab``
Structure Tags          ``ui.tidy_panel.build_tidy_panel``
======================  =========================================

That is the point: the panel is a *layout*, so the tidy logic, the tests that
pin it and the ``manager.<name>`` attribute contract all keep working. A second
implementation of the tidy pass is the bug this shape avoids.

BUILD IT ONCE
-------------
``build_lyrics_tab`` and ``build_tidy_panel`` write their widgets onto the
manager as ``manager.<name>``. Calling either of them a second time therefore
rebuilds those attributes and orphans the first set — the orphaned widgets stay
visible and connected, but the handlers' ``manager.<name>`` lookups now resolve
to the new ones, so the two copies drift. ``build_lyrics_panel`` must be called
in place of both, never as well as them.

Follows the ``ui/settings_tab.py`` / ``ui/lyrics_tab.py`` pattern: every widget
is stored as ``manager.<name>`` so handlers can read them back.
"""
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

#: The inner tab titles, in build order. Exposed as a tuple so a test (or the
#: manager) can name a tab instead of hardcoding an index — index-based tab
#: lookups are how the wrong panel ends up refreshing when one is inserted.
PANEL_TABS = ("Normalize & Tidy", "Structure Tags")


def _budget_line(manager):
    """Create the panel's VIEW of the token budget, and store it on the manager.

    Deliberately a separate label, not the manager's own
    ``lyrics_limit_label`` moved here:

    * that object is placed under the waveform by ``DatasetManager.init_ui``
      (line order: the panel is built long before it), and moving it would take
      the counter away from the panel you type lyrics into; and
    * a panel that *depends* on construction order is a trap — building the
      panel before the label would silently orphan it, which is exactly what
      happened the first time this was written.

    There is still only ONE counter: ``update_lyric_limit_label`` is the sole
    writer and writes both views from one measurement (see the note there), so
    they cannot disagree.
    """
    label = QLabel("")
    label.setWordWrap(True)
    label.setTextFormat(Qt.RichText)
    label.setToolTip(
        "How much of ACE-Step's token cap these lyrics use. Counted with the "
        "Qwen3-Embedding-0.6B tokenizer ACE-Step itself uses; “≈” means the "
        "exact tokenizer is not installed and the number is an estimate."
    )
    manager.lyrics_panel_limit_label = label
    return label


def _open_limits(manager):
    """Open the token-cap explanation dialog (``ui.lyric_limit_notice``)."""
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    return acknowledge_lyrics_limit(manager)


def refresh_panel_summary(manager):
    """Rewrite the footer's "rules in effect" line.

    The line exists because the tidy options are set on a different half of the
    page: without it you would have to scan the tab to find out what a tidy pass
    is about to do. ``rule_summary`` reads the same ``_lyrics_options()`` the
    pass reads, so the two cannot drift.
    """
    label = getattr(manager, "lyrics_panel_summary", None)
    if label is None:
        return
    from ui.lyrics_tidy_controls import rule_summary

    label.setText(
        "Tidy rules in effect: " + rule_summary(manager)
        + ". Change them on the “Normalize & Tidy” tab."
    )


def build_lyrics_panel(manager, parent):
    """Build the unified Lyrics panel, parented to ``parent``, and return it.

    Call this INSTEAD of ``build_lyrics_tab`` and ``build_tidy_panel`` — see
    the module docstring's "BUILD IT ONCE" section for why building the panel
    and either of its parts corrupts the ``manager.<name>`` contract.

    ``parent`` must be a bare ``QWidget``: ``build_lyrics_tab`` sets its
    ``QVBoxLayout`` directly on whatever it is handed.
    """
    panel = QWidget(parent)
    panel.setObjectName("lyricsPanel")
    root = QVBoxLayout(panel)
    root.setContentsMargins(0, 0, 0, 0)
    root.setSpacing(0)

    tabs = QTabWidget(panel)
    root.addWidget(tabs, 1)

    # --- 1. Normalize & Tidy: the whole of the old 🎵 Lyrics tab -------------
    from ui.lyrics_tab import build_lyrics_tab

    tidy_page = QWidget()
    build_lyrics_tab(manager, tidy_page)
    tabs.addTab(tidy_page, PANEL_TABS[0])

    # --- 2. Structure Tags: the old Tags & checks group, unchanged ----------
    from ui.tidy_panel import build_tidy_panel

    tags_page = QWidget()
    tags_layout = QVBoxLayout(tags_page)
    tags_layout.setContentsMargins(10, 8, 10, 8)
    tags_layout.addWidget(build_tidy_panel(manager, tags_page))
    tags_layout.addStretch()
    tabs.addTab(tags_page, PANEL_TABS[1])

    # --- Footer: what the rules will do, how many tokens that leaves --------
    # The two numbers that matter most belong beside the rules, always in view.
    footer = QVBoxLayout()
    footer.setContentsMargins(10, 6, 10, 6)
    footer.setSpacing(4)

    manager.lyrics_panel_summary = QLabel("")
    manager.lyrics_panel_summary.setProperty("muted", True)
    manager.lyrics_panel_summary.setWordWrap(True)
    footer.addWidget(manager.lyrics_panel_summary)

    footer.addWidget(_budget_line(manager))

    limits_row = QHBoxLayout()
    manager.lyrics_panel_limits_btn = QPushButton("ℹ Token limits…")
    manager.lyrics_panel_limits_btn.setToolTip(
        "How many tokens ACE-Step keeps for lyrics (512 training / 2048 "
        "inference) and for the caption (256), and why the counts here say "
        "“≈” when the exact tokenizer is not installed."
    )
    limits_row.addStretch()
    limits_row.addWidget(manager.lyrics_panel_limits_btn)
    footer.addLayout(limits_row)
    root.addLayout(footer)

    manager.lyrics_panel_limits_btn.clicked.connect(
        lambda: _open_limits(manager))

    manager.lyrics_panel = panel
    manager.lyrics_panel_tabs = tabs
    # The summary describes options owned by tab 0, so refresh it on every
    # switch rather than only at build time.
    tabs.currentChanged.connect(lambda _i: refresh_panel_summary(manager))
    refresh_panel_summary(manager)
    return panel
