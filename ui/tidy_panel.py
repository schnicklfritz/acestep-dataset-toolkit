"""Lyrics Tidy panel — the three structure-tag rules, on the Tags && checks tab.

The rest of the tidy options live on the Lyrics tab next to the contraction
table. These three are here instead because they are *tag* rules: they fix the
structure markers that the Tag Manager and the audit tools already populate this
page with, and because the panel offers both scopes the user asked for —

  * ``Tidy Selected Track`` — the row currently selected in the Dataset Studio;
  * ``Tidy Entire Dataset`` — every track, with a count and a confirmation.

The rules are small, blocking string passes over text already in memory, so they
run inline. A worker and a progress bar would be more code than work: the passes
are line-local, and the confirmation dialog is the only thing standing between a
click and the dataset.

Follows the ``ui/lyrics_tab.py`` pattern: ``build_tidy_panel(manager, parent)``
stores every widget as ``manager.<name>`` so the handlers can read them back.
"""
from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


def _options(manager):
    """The three rules, as ``normalize_lyrics`` keyword arguments.

    Named after the checkbox attributes rather than the pass names so the two
    cannot drift: the checkbox IS the option.
    """
    return {
        "do_strip_quotes": manager.lyrics_quotes_check.isChecked(),
        "do_trim_tag_modifiers": manager.lyrics_tagtrim_check.isChecked(),
        "do_drop_stray_tags": manager.lyrics_tagdrop_check.isChecked(),
    }


def _report_line(report):
    """One-line summary of a tidy report. Zero-count parts are omitted."""
    bits = []
    if report.get("quotes"):
        bits.append(f"{report['quotes']} quote(s) stripped")
    if report.get("tags_trimmed"):
        bits.append(f"{report['tags_trimmed']} tag(s) trimmed")
    if report.get("tags_dropped"):
        bits.append(f"{report['tags_dropped']} stray tag line(s) dropped")
    if report.get("tags"):
        bits.append(f"{report['tags']} tag(s) capitalised")
    if report.get("lines_changed"):
        bits.append(f"{report['lines_changed']} line(s) changed")
    return ", ".join(bits) if bits else "no changes"


def _source_of(sample):
    """The text a tidy pass should start from.

    ``raw_lyrics`` first: it is the pre-tidy text, so re-running the tidy stays
    idempotent and the original survives. Same order the Lyrics tab uses.
    """
    return (sample.get("raw_lyrics") or sample.get("formatted_lyrics")
            or sample.get("lyrics") or "")


def _tidy_one(sample, opts):
    """Work out one track's tidied text. Returns ``(new_text, report)`` or ``None``.

    Deliberately PURE — it computes, it does not write. The caller has to take
    the per-song backup BETWEEN this call and the write, so a function that
    mutated the sample here would have the backup capturing the already-tidied
    text, which is exactly the state the backup exists to preserve. ``None``
    means the track had nothing to tidy or the text did not change.
    """
    from modules.lyrics_normalizer import normalize_lyrics

    source = _source_of(sample)
    if not source.strip():
        return None
    new_text, report = normalize_lyrics(source, **opts)
    if new_text == source:
        return None
    return new_text, report


def _apply_tidy_to(sample, new_text):
    """Write the tidied text into every field the dataset contract names.

    ``formatted_lyrics`` is what the exporter reads, ``lyrics`` is what the
    inspector shows, and ``raw_lyrics`` keeps the PRE-tidy text so the run stays
    reversible and re-runnable.
    """
    if not sample.get("raw_lyrics"):
        sample["raw_lyrics"] = _source_of(sample)
    sample["formatted_lyrics"] = new_text
    sample["lyrics"] = new_text


def _write_lyrics_sidecar(manager, sample):
    """Write this track's lyrics next to its audio as ``<stem>_lyrics.txt``.

    Underscored deliberately: ``modules.exporters.export_sidecar_captions``
    already owns ``<stem>.txt`` for the caption, so a bare ``<stem>.txt`` here
    would be silently overwritten on the next caption export. Silently skipped
    when the track has no audio path on disk — a tidy must not fail because the
    file moved.
    """
    import os

    path = sample.get("audio_path") or ""
    if not path or not os.path.exists(path):
        return None
    text = (sample.get("formatted_lyrics") or sample.get("lyrics")
            or sample.get("raw_lyrics") or "")
    if not text.strip():
        return None
    root, _ext = os.path.splitext(path)
    dest = f"{root}_lyrics.txt"
    try:
        with open(dest, "w", encoding="utf-8") as f:
            f.write(text.rstrip() + "\n")
    except OSError:
        return None
    return dest


def apply_tidy(manager, scope):
    """Run the three tag rules over ``scope`` ('selected' or 'dataset').

    Returns the number of tracks changed. Both buttons route through here so the
    confirmation, the snapshot and the backup cannot diverge between them.
    """
    if scope == "selected":
        sample = manager.get_selected_sample()
        targets = [sample] if sample else []
        if not targets:
            QMessageBox.warning(
                manager, "No Track Selected",
                "Select a track in the Dataset Studio table first.",
            )
            return 0
    else:
        targets = list(manager.dataset.get("samples", []))
        if not targets:
            QMessageBox.warning(manager, "No Tracks", "Add audio tracks first.")
            return 0
        confirm = QMessageBox(manager)
        confirm.setWindowTitle("Tidy Entire Dataset")
        confirm.setIcon(QMessageBox.Warning)
        confirm.setText(
            f"Run the tag tidy over <b>all {len(targets)} track(s)</b>?"
        )
        confirm.setInformativeText(
            "Each track's lyrics are rewritten in place: quotes stripped, tag "
            "modifiers trimmed, stacked tags dropped.<br><br>"
            "The run is undoable, and each track is backed up once to "
            "<code>_Backup/songs/</code> the first time it changes."
        )
        confirm.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
        confirm.setDefaultButton(QMessageBox.Cancel)  # deliberate: default is cancel
        if confirm.exec() != QMessageBox.Yes:
            manager.status_label.setText("Tidy cancelled; nothing changed.")
            return 0

    opts = _options(manager)
    if not any(opts.values()):
        QMessageBox.information(
            manager, "No Rules Ticked",
            "Tick at least one tidy rule, then press the button again.",
        )
        return 0

    manager.record_snapshot()
    totals = {"quotes": 0, "tags_trimmed": 0, "tags_dropped": 0, "tags": 0,
              "lines_changed": 0}
    changed = 0
    for sample in targets:
        result = _tidy_one(sample, opts)
        if result is None:
            continue
        new_text, report = result
        # Back up BEFORE the write, and only for a track that is actually about
        # to change: a backup taken after the edit would preserve the tidied
        # text, which is the one version you can already reconstruct.
        backup_song = getattr(manager, "_backup_song_once", None)
        if backup_song is not None:
            backup_song(sample)
        _apply_tidy_to(sample, new_text)
        _write_lyrics_sidecar(manager, sample)
        for key in totals:
            totals[key] += report.get(key, 0) or 0
        changed += 1

    manager.refresh_table()
    manager.on_table_selection_changed()
    label = "selected track" if scope == "selected" else f"{changed} track(s)"
    manager.status_label.setText(f"Tags tidied on {label}: " + _report_line(totals))
    return changed


def build_tidy_panel(manager, parent):
    """Build the 'Lyrics Tidy' group, parented to ``parent``, and return it."""
    grp = QGroupBox("Lyrics Tidy — structure tags", parent)
    layout = QVBoxLayout(grp)

    hint = QLabel(
        "Three fixes for the structure tags that come back from transcription "
        "and from the tag tools. They run over one track or the whole dataset."
    )
    hint.setProperty("muted", True)
    hint.setWordWrap(True)
    layout.addWidget(hint)

    manager.lyrics_quotes_check = QCheckBox(
        'Strip every double quote  (“hi” → hi)'
    )
    manager.lyrics_quotes_check.setChecked(True)
    manager.lyrics_quotes_check.setToolTip(
        "A quote is not a lyric and not a delimiter: the model does not sing it, "
        "and a line that opens with one is skipped by the capitalizer. "
        "Apostrophes are a different character and are left to the contraction "
        "table."
    )
    layout.addWidget(manager.lyrics_quotes_check)

    manager.lyrics_tagtrim_check = QCheckBox(
        "Trim modifiers inside tags  ([Chorus - Raspy Vocals] → [Chorus])"
    )
    manager.lyrics_tagtrim_check.setChecked(True)
    manager.lyrics_tagtrim_check.setToolTip(
        "Cuts everything from the first ' -' inside a bracket. A joined hyphen is "
        "not a separator, so [Pre-Chorus] is left alone. A bracket that starts "
        "with a two-letter language code ([EN - Verse], [JA - Verse - sparse]) is "
        "left verbatim, because there the ' -' separates language from section."
    )
    layout.addWidget(manager.lyrics_tagtrim_check)

    manager.lyrics_tagdrop_check = QCheckBox(
        "Drop stacked tags  ([Verse 1] then [Raspy Vocal] → keep [Verse 1])"
    )
    manager.lyrics_tagdrop_check.setChecked(True)
    manager.lyrics_tagdrop_check.setToolTip(
        "When two tag-only lines sit next to each other, only the first survives. "
        "A tag after lyric text, or after a blank line, is never dropped."
    )
    layout.addWidget(manager.lyrics_tagdrop_check)

    row = QHBoxLayout()
    manager.tidy_selected_btn = QPushButton("✨ Tidy Selected Track")
    manager.tidy_selected_btn.setToolTip(
        "Apply the ticked rules to the track selected in the Dataset Studio "
        "(undoable)."
    )
    manager.tidy_dataset_btn = QPushButton("✨ Tidy Entire Dataset")
    manager.tidy_dataset_btn.setToolTip(
        "Apply the ticked rules to every loaded track, after a confirmation. "
        "Undoable, and each track is backed up once."
    )
    row.addWidget(manager.tidy_selected_btn)
    row.addWidget(manager.tidy_dataset_btn)
    row.addStretch()
    layout.addLayout(row)

    manager.tidy_selected_btn.clicked.connect(
        lambda: apply_tidy(manager, "selected"))
    manager.tidy_dataset_btn.clicked.connect(
        lambda: apply_tidy(manager, "dataset"))
    return grp
