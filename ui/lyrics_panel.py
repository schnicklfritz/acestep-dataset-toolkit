"""Unified Lyrics panel — the one surface every lyrics tool now lives on.

What this gathers
-----------------
Four lyrics surfaces used to be reachable from four unrelated places:

* the **🎵 Lyrics tab** — ``ui.lyrics_tab.build_lyrics_tab`` — the tidy
  options, the before/after preview, the editable contraction table and the
  whole-dataset "All Lyrics" block;
* the **"Lyrics Tidy — structure tags" group** — ``ui.tidy_panel`` — which
  lived at the bottom of the *Tags & checks* page, so the three tag rules were
  nowhere near the other lyric rules they are always used with;
* the **token-budget line** — ``manager.lyrics_limit_label``, which sat under
  the waveform and whose "full explanation" was only reachable from a tooltip;
* the **Lyrics Studio window** — ``ui.lyrics_studio`` — which the Lyrics tab's
  own two buttons (``Edit Raw Lyrics…`` and ``Lyric Capitalizer…``) opened.

They are one panel now. The two tidy surfaces become the first two inner tabs,
so the tag rules and the contraction rules are visibly the same job; the
Studio tab carries the budget line, a plain-language summary of the rules in
effect, and the button that opens the editor window.

What this module deliberately does NOT do
-----------------------------------------
It does not reimplement anything. Every widget is built by the module that
already owned it and every action calls the handler that already existed:

======================  =========================================
Inner tab               Built by / wired to
======================  =========================================
Normalize & Tidy        ``ui.lyrics_tab.build_lyrics_tab``
Structure Tags          ``ui.tidy_panel.build_tidy_panel``
Studio                  ``ui.lyrics_studio`` + ``manager`` handlers
======================  =========================================

That is the point: the panel is a *layout*, so the tidy logic, the tests that
pin it and the ``manager.<name>`` attribute contract all keep working. A
second implementation of the tidy pass is the bug this shape avoids.

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
PANEL_TABS = ("Normalize & Tidy", "Structure Tags", "Studio")


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


def _open_studio(manager):
    """Open the editor window, preferring the manager's own entry point.

    ``manager.open_lyrics_editor`` is the established seam (it is what the
    Lyrics tab's button called, and what ``.agent_notes`` pins by name), so it
    wins when it exists. The lazy import is the fallback for a manager built by
    a test or by tooling that has no handler.
    """
    opener = getattr(manager, "open_lyrics_editor", None)
    if callable(opener):
        return opener()
    from ui.lyrics_studio import open_lyrics_studio

    return open_lyrics_studio(manager)


def _open_limits(manager):
    """Open the token-cap explanation dialog (``ui.lyric_limit_notice``)."""
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    return acknowledge_lyrics_limit(manager)


def refresh_panel_summary(manager):
    """Rewrite the Studio tab's "rules in effect" line.

    The line exists because the tidy options are set on a *different* inner
    tab: without it you would have to switch tabs to find out what the Studio's
    ✨ Tidy button is about to do.
    """
    label = getattr(manager, "lyrics_panel_summary", None)
    if label is None:
        return
    from ui.lyrics_studio import rule_summary

    label.setText(
        "Tidy rules in effect: " + rule_summary(manager)
        + ". Change them on the “Normalize & Tidy” tab."
    )


def _studio_page(manager, parent):
    """The Studio tab: budget line, rule summary, editor button, cap help."""
    page = QWidget(parent)
    lay = QVBoxLayout(page)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(10)

    # The budget line, first: it is the number you check before writing a song
    # back into a track, and it is the one thing this tab owns outright.
    lay.addWidget(_budget_line(manager))

    hint = QLabel(
        "The editor is where a song is pasted, its shouted words are marked, and "
        "the result is written back to the track. It opens in its own window so "
        "a full song fits on screen."
    )
    hint.setProperty("muted", True)
    hint.setWordWrap(True)
    lay.addWidget(hint)

    gesture = QLabel(
        "<b>Double-click a word to UPPERCASE it</b> (that is the belting / "
        "shouted delivery mark); double-click it again to undo. The same "
        "gesture works in the Track Inspector's inline lyrics field, because "
        "both use the one editor widget."
    )
    gesture.setWordWrap(True)
    lay.addWidget(gesture)

    manager.lyrics_panel_summary = QLabel("")
    manager.lyrics_panel_summary.setProperty("muted", True)
    manager.lyrics_panel_summary.setWordWrap(True)
    lay.addWidget(manager.lyrics_panel_summary)

    row = QHBoxLayout()
    manager.lyrics_panel_studio_btn = QPushButton("🔠 Open Lyrics Studio…")
    manager.lyrics_panel_studio_btn.setToolTip(
        "Edit the selected track's lyrics: paste, double-click a word to mark "
        "it shouted, review the diff against the track, then write it back."
    )
    manager.lyrics_panel_limits_btn = QPushButton("ℹ Token limits…")
    manager.lyrics_panel_limits_btn.setToolTip(
        "How many tokens ACE-Step keeps for lyrics (512 training / 2048 "
        "inference) and for the caption (256), and why the counts here say "
        "“≈” when the exact tokenizer is not installed."
    )
    row.addWidget(manager.lyrics_panel_studio_btn)
    row.addWidget(manager.lyrics_panel_limits_btn)
    row.addStretch()
    lay.addLayout(row)

    lay.addStretch()

    manager.lyrics_panel_studio_btn.clicked.connect(
        lambda: _open_studio(manager))
    manager.lyrics_panel_limits_btn.clicked.connect(
        lambda: _open_limits(manager))

    refresh_panel_summary(manager)
    return page


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

    # --- 3. Studio: budget line + the editor's entry points -----------------
    tabs.addTab(_studio_page(manager, tabs), PANEL_TABS[2])

    # The summary line describes options owned by tab 0, so refresh it on every
    # switch rather than only at build time.
    tabs.currentChanged.connect(lambda _i: refresh_panel_summary(manager))

    manager.lyrics_panel = panel
    manager.lyrics_panel_tabs = tabs
    return panel
