"""The tidy rules, declared once and shown where the lyrics are edited.

Why this module exists
----------------------
The four rule checkboxes used to live on exactly one surface: the ``Tidy
Options`` group inside the Tools window's Lyrics tab -- which is itself inside
a ``QScrollArea``. The owner's report was that the rules were therefore not
visible while editing, and that "Tidy" in the Lyrics Studio applied rules you
could not see or change. Moving the *widgets* is not possible: the checkboxes
are the app's tidy state, and two tests pin them inside that panel
(``tests/test_lyrics_panel.py`` ``isAncestorOf(manager.lyrics_lines_check)``).

So the state stays where it is and this module gives the Studio a SECOND VIEW
of it:

* the wording of every rule lives HERE (``RULE_SPECS``) and ``ui/lyrics_tab.py``
  builds its checkboxes from it, so the two surfaces cannot describe the same
  rule differently;
* :class:`TidyRulesPanel` is a compact, always-visible group of the same four
  ticks. Each tick drives the manager's own checkbox, so there is still exactly
  ONE piece of tidy state and ``manager._lyrics_options()`` is unchanged --
  the Lyrics tab, the tag panel (``ui/tidy_panel.py``), the dataset-wide apply
  and the Studio all read the same values;
* the reverse direction is wired too: changing a tick on the Lyrics tab while
  the Studio is open refreshes the Studio's view of it.

A manager with no tab widgets (a test stand-in, tooling) still gets a working
panel: the ticks then stand alone and are used for the editor-buffer tidy only,
which is what :meth:`TidyRulesPanel.options` documents.
"""

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

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
    return boxes


def _fallback_options(ticks):
    """Options for a manager with no ``_lyrics_options`` (test stand-in, tooling)."""
    from modules.lyrics_normalizer import (
        DEFAULT_CASE_FIXED,
        DEFAULT_CONTRACTIONS,
        DEFAULT_ING_EXCEPTIONS,
    )

    return {
        "contractions": dict(DEFAULT_CONTRACTIONS),
        "ing_to_in": ticks["ing_to_in"],
        "ing_exceptions": set(DEFAULT_ING_EXCEPTIONS),
        "case_fixed": set(DEFAULT_CASE_FIXED),
        "do_capitalize_tags": ticks["do_capitalize_tags"],
        "do_capitalize_lines": ticks["do_capitalize_lines"],
        "do_strip_punctuation": ticks["do_strip_punctuation"],
    }



class TidyRulesPanel(QGroupBox):
    """The four tidy rules, always visible, beside the text they apply to.

    ``with_scope`` adds the "apply to" scope picker and the dataset-wide apply
    button, which is what makes the Studio a complete tidy surface rather than
    only a buffer-tidy one.

    The panel holds NO tidy state of its own: every tick is pushed onto the
    manager's checkbox (the app's single source) and pulled back from it, so the
    value the Lyrics tab shows and the value this panel shows cannot disagree.
    """

    TITLE = "Tidy rules \u2014 always visible"

    def __init__(self, manager, *, with_scope=False, on_apply=None, parent=None):
        super().__init__(self.TITLE, parent)
        self.manager = manager
        # Re-entrancy guard: a tick here sets the manager's checkbox, whose
        # ``stateChanged`` calls back into ``refresh``. Without this the two
        # would ping-pong (harmlessly today, but it makes the direction of a
        # change unreadable and would bite the first time a rule is derived).
        self._syncing = False
        self.boxes = {}
        self._bound = {}

        grid = QGridLayout(self)
        grid.setContentsMargins(8, 6, 8, 6)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(2)

        for spec in RULE_SPECS:
            key = spec["key"]
            box = QCheckBox(spec["label"])
            box.setToolTip(spec["tip"])
            box.setChecked(spec["default"])
            self.boxes[key] = box
            tab_widget = getattr(manager, spec["attr"], None)
            if tab_widget is not None and hasattr(tab_widget, "isChecked"):
                # Bound to the manager's own checkbox: copying its state in, and
                # following it afterwards. A real manager always has it (the tab
                # is built before any window); a stand-in may not.
                self._bound[key] = tab_widget
                box.setChecked(tab_widget.isChecked())
                box.toggled.connect(lambda on, k=key: self._push(k, on))
                tab_widget.stateChanged.connect(lambda *_: self.refresh())
            # An unbound tick is simply this panel's own; nothing else reads it.

        for row, spec in enumerate(RULE_SPECS):
            grid.addWidget(self.boxes[spec["key"]], row // 2, row % 2)

        self.scope_label = None
        self.scope_combo = None
        self.apply_btn = None
        if with_scope:
            scope = QHBoxLayout()
            self.scope_label = QLabel("Apply to:")
            self.scope_combo = QComboBox()
            self.scope_combo.addItems(list(SCOPE_CHOICES))
            self.scope_combo.setToolTip(
                "Which tracks a dataset-wide tidy rewrites. Instrumentals are "
                "skipped by the last option because they have no lyrics."
            )
            self.apply_btn = QPushButton("\u2705 Apply to dataset")
            self.apply_btn.setToolTip(
                "Run these rules over the tracks in scope and save. Undoable "
                "(Ctrl+Z on the main window)."
            )
            if callable(on_apply):
                self.apply_btn.clicked.connect(on_apply)
            else:
                self.apply_btn.setEnabled(False)
            scope.addWidget(self.scope_label)
            scope.addWidget(self.scope_combo)
            scope.addWidget(self.apply_btn)
            scope.addStretch()
            grid.addLayout(scope, (len(RULE_SPECS) + 1) // 2, 0, 1, 2)
        self._grid = grid

    # ---- state ----------------------------------------------------------
    def ticks(self):
        """The four rules as this panel currently shows them."""
        return {key: box.isChecked() for key, box in self.boxes.items()}

    def scope_text(self):
        """The selected scope, or the default when this panel has no picker."""
        if self.scope_combo is None:
            return SCOPE_CHOICES[0]
        return self.scope_combo.currentText()

    def options(self):
        """The effective tidy options, ready for ``normalize_lyrics(**opts)``.

        Prefers the manager's own ``_lyrics_options()`` -- it also merges the
        contraction table, the profile and the tag rules, and it reads the very
        checkboxes this panel drives, so the two agree by construction. The
        fallback exists only for a manager that has no such method, and is
        documented as such rather than being a second rule set.
        """
        getter = getattr(self.manager, "_lyrics_options", None)
        if callable(getter):
            try:
                return dict(getter())
            except Exception:  # noqa: BLE001 -- a stand-in may raise; use the ticks
                pass
        return _fallback_options(self.ticks())

    def refresh(self):
        """Re-read the bound ticks, so a change on the Lyrics tab shows here."""
        if self._syncing:
            return
        self._syncing = True
        try:
            for key, widget in self._bound.items():
                self.boxes[key].setChecked(widget.isChecked())
        finally:
            self._syncing = False

    # ---- wiring ---------------------------------------------------------
    def _push(self, key, checked):
        """Send a tick to the manager's checkbox (the app's tidy state)."""
        if self._syncing:
            return
        widget = self._bound.get(key)
        if widget is None:
            return
        self._syncing = True
        try:
            widget.setChecked(bool(checked))
        finally:
            self._syncing = False

    # ---- Qt -------------------------------------------------------------
    def showEvent(self, event):
        # The window is cached and re-shown for other tracks, and the Lyrics tab
        # may have been edited in between, so re-read before displaying.
        self.refresh()
        super().showEvent(event)
