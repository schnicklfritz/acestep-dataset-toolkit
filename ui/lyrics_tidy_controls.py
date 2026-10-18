"""The tidy rules, declared once, for the Lyrics page and its summary line.

The four rule checkboxes are the app's tidy state. They are built by
:func:`build_rule_checkboxes`, which ``ui/lyrics_tab.py`` calls, and the wording
lives here in ``RULE_SPECS`` so no second surface can describe the same rule
differently.

The checkboxes on the "Normalize & Tidy" tab are the only view of these rules,
and :func:`rule_summary` gives the panel's one-line "rules in effect" readout
from the same state.

:func:`rule_summary` reads ``manager._lyrics_options()`` -- the manager's own
merge of the ticks, the contraction table, the profile and the tag rules -- so
the summary cannot disagree with what a tidy pass will actually do.
"""

from PySide6.QtWidgets import QCheckBox

# ---------------------------------------------------------------------------
# The rules, declared once.
# ---------------------------------------------------------------------------
# ``attr``  -- the name the widget is stored under on the manager. Unchanged
#              from when this module's caller (``ui/lyrics_tab.py``) built them,
#              because ``_lyrics_options()`` and two test modules name them.
# ``key``   -- the key in the options dict ``modules.lyrics_normalizer`` takes.
# ``label`` / ``tip`` -- the wording, identical on both surfaces by construction.
RULE_SPECS = (
    {
        "attr": "lyrics_ing_check",
        "key": "ing_to_in",
        "default": False,
        "label": "Shorten -ing to -in  (running \u2192 runnin)",
        "tip": (
            "Singer-dependent: Ozzy Osbourne enunciates '-ing', many singers do "
            "not. Off by default; turn it on to match how the track is actually "
            "sung."
        ),
    },
    {
        "attr": "lyrics_tags_check",
        "key": "do_capitalize_tags",
        "default": True,
        "label": "Capitalize structure tags  ([verse] \u2192 [Verse])",
        "tip": "Structure tags are markers, not words, so they are cased to match.",
    },
    {
        "attr": "lyrics_lines_check",
        "key": "do_capitalize_lines",
        "default": True,
        "label": "Capitalize the first word of each line  (rising \u2192 Rising)",
        "tip": (
            "Lyric lines are normally written as sentences, so a lowercase start "
            "reads as a typo in a training caption. Only the first character is "
            "touched \u2014 the rest of the line keeps its case, so an ALL-CAPS "
            "line and words like iPhone/eBay are left alone."
        ),
    },
    {
        "attr": "lyrics_punct_check",
        "key": "do_strip_punctuation",
        "default": True,
        "label": 'Strip trailing punctuation  (except " and ))',
        "tip": (
            "Removes . , ! ? ; : - \u2026 from the end of each lyric line. A "
            'closing quote " or paren ) is preserved.'
        ),
    },
)

# The scope choices, spelled exactly as ``ui/lyrics_tab.py`` spells them:
# ``DatasetManager._lyrics_scope_samples`` matches on these prefixes.
SCOPE_CHOICES = ("Selected Track", "All Tracks", "All Tracks (instrumentals excluded)")


def build_rule_checkboxes(manager, form):
    """Create the four checkboxes ON ``manager``, as ``lyrics_tab`` has always done.

    Kept here so the labels and tooltips have a single source: the tab passes its
    ``QFormLayout`` and gets back the widgets already laid out and defaulted.

    The four assignments at the end are written out longhand on purpose. The loop
    above cannot name the attributes it sets (``spec["attr"]`` is a computed
    string), and ``tests/test_dangling_attributes.py`` — the guard written after
    the ``run_ai_btn`` bug, where a removed widget left ``self.run_ai_btn``
    references behind — only sees a ``setattr`` whose name is a string LITERAL.
    It caught this: with only the loop, ``_lyrics_options()``'s reads of
    ``self.lyrics_punct_check`` and friends looked dangling. An ALLOWLIST entry
    would have blinded the guard for exactly the bug it exists to catch, so the
    names are spelled out here instead.
    """
    boxes = {}
    for spec in RULE_SPECS:
        box = QCheckBox(spec["label"])
        box.setToolTip(spec["tip"])
        box.setChecked(spec["default"])
        form.addRow("", box)
        boxes[spec["key"]] = box

    manager.lyrics_ing_check = boxes["ing_to_in"]
    manager.lyrics_tags_check = boxes["do_capitalize_tags"]
    manager.lyrics_lines_check = boxes["do_capitalize_lines"]
    manager.lyrics_punct_check = boxes["do_strip_punctuation"]


def rule_summary(manager):
    """One line describing the tidy options in effect, so they cannot be guessed."""
    try:
        opts = manager._lyrics_options()
    except Exception:  # noqa: BLE001 — the summary must never break the panel
        return "unavailable"
    on = []
    if opts.get("ing_to_in"):
        on.append("-ing→-in")
    if opts.get("do_capitalize_tags"):
        on.append("tags")
    if opts.get("do_capitalize_lines"):
        on.append("line starts")
    if opts.get("do_strip_punctuation"):
        on.append("punctuation")
    count = len(opts.get("contractions") or {})
    return (
        (", ".join(on) if on else "no optional rules")
        + f", {count} contraction(s)"
    )



