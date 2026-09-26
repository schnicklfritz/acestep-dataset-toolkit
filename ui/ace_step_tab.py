"""ACE-Step (Kaggle) tab builder for DatasetManager.

The ACE-Step captioner is the one backend with a whole run pipeline around it:
stage the dataset's audio locally as MP3, upload it as a private Kaggle dataset,
push the caption kernel, download the captions into a folder the user picked,
then diff them against the captions already in the dataset. Those controls used
to be scattered through the shared 🎤 Caption tab, where they competed for space
with four other backends (Gemini, DeepSeek, custom endpoint, MOSS).

One STRIP, in the order the steps happen
----------------------------------------
Tick tracks -> Stage -> Caption on Kaggle -> Review the diff. Every action sits on
ONE row in that order, because the page used to be five groups stacked by KIND of
thing (credentials, folders, prompt, staging, run) with the track selector at the
BOTTOM and "add the ticked tracks" ABOVE it — so it asked the user to tick below
and stage above, and scattered four status labels across the page.

Only the next action is live at a time and a disabled one says WHY in its tooltip
(``DatasetManager.update_ace_actions``); everything set-once is collapsed into
⚙ Settings, so the strip is what the page looks like.
``build_ace_step_tab(manager, parent)``, which builds widgets directly onto
``manager`` and stores them as ``manager.<name>``, so DatasetManager methods
(save_pipeline_defaults, start_ai_captioning, ...) keep working by name.

The three settings that make the run reproducible are here and nowhere else:
  * ``caption_prompt_addendum`` — extra text APPENDED to the caption prompt;
  * ``caption_staging_dir``     — the local folder that IS the uploaded dataset;
  * ``caption_output_dir``      — the local folder the captions come back to.
"""
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from modules.caption_kaggle_run import default_output_dir, default_staging_dir
# The same tickable-dropdown used by the MOSS row: a menu with "Select all /
# Select none / Select tracks missing captions" plus a checkbox per track, with
# ticks keyed by FILENAME so they survive table refreshes and reloads.
from ui.caption_tab import TrackPickerButton


def _spin(low, high, value):
    """Small guarded spin box (see modules/wheel_guard.py)."""
    from modules.wheel_guard import GuardedSpinBox

    box = GuardedSpinBox()
    box.setRange(low, high)
    box.setValue(value)
    return box


def _folder_row(edit, button, label):
    """A 'label: [path] [Browse…]' row, shared by both folder settings."""
    row = QHBoxLayout()
    row.addWidget(QLabel(label))
    row.addWidget(edit, 1)
    row.addWidget(button)
    return row


def _head(text):
    """A small bold caption inside ⚙ Settings, in place of a nested group box."""
    return QLabel(f"<b>{text}</b>")


def build_ace_step_tab(manager, parent):
    outer = QVBoxLayout(parent)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(0)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.NoFrame)
    outer.addWidget(scroll)

    inner = QWidget()
    layout = QVBoxLayout(inner)
    layout.setContentsMargins(10, 8, 10, 8)
    layout.setSpacing(10)
    scroll.setWidget(inner)

    hint = QLabel(
        "Tick tracks → <b>Stage</b> → <b>Caption</b> on a free Kaggle GPU → "
        "<b>Review</b> the diff and choose what to keep. Folders, dataset, prompt "
        "and limits live in ⚙ Settings at the bottom."
    )
    hint.setProperty("muted", True)
    hint.setWordWrap(True)
    layout.addWidget(hint)

    # ------------------------------------------------------------------
    # THE STEP STRIP — one row, left to right, in the order the steps happen.
    # The widgets are ADDED to it further down, once they all exist, so the order
    # on the strip is the WORKFLOW order and not the creation order of this file.
    # It WRAPS, left to right, so the whole sequence stays visible in the
    # narrow Tools panel (it used to scroll sideways, which hid most of it).
    # ------------------------------------------------------------------
    from ui.shell import FlowLayout

    from ui.shell import height_for_width_policy

    manager.ace_strip = QWidget()
    manager.ace_strip.setSizePolicy(height_for_width_policy())
    strip = FlowLayout(manager.ace_strip)
    layout.addWidget(manager.ace_strip)

    # THE one status line: what the run is doing right now. It used to be four
    # labels in four places (credentials, staging count, tick count, run).
    manager.ace_status_label = QLabel("Not run yet.")
    manager.ace_status_label.setProperty("muted", True)
    manager.ace_status_label.setWordWrap(True)
    layout.addWidget(manager.ace_status_label)

    # Everything set-once, COLLAPSED, so the strip is what the page is. Created
    # here and added to `layout` after the staging group — a widget's position
    # follows the addWidget call, not the order it was constructed in.
    settings_grp = QGroupBox("⚙ Settings — folders · dataset · prompt · limits")
    settings_grp.setCheckable(True)
    settings_grp.setChecked(False)
    settings_layout = QVBoxLayout(settings_grp)

    # ------------------------------------------------------------------
    # Kaggle credentials (stored, or session-only, or not set yet)
    # ------------------------------------------------------------------
    settings_layout.addWidget(_head("Kaggle credentials"))
    c_layout = QVBoxLayout()
    settings_layout.addLayout(c_layout)

    manager.ace_cred_status = QLabel("Kaggle credentials: checking…")
    manager.ace_cred_status.setWordWrap(True)
    manager.ace_cred_status.setToolTip(
        "Where the Kaggle API key currently lives.\n\n"
        "• stored     — in the OS keyring (or the encrypted secrets file), so runs "
        "never ask again\n"
        "• this session only — you asked NOT to store it; nothing was written to "
        "disk and it must be re-entered after a restart\n"
        "• not set    — a run will ask before doing anything else"
    )
    c_layout.addWidget(manager.ace_cred_status)

    c_row = QHBoxLayout()
    manager.ace_cred_test_btn = QPushButton("🔌 Kaggle")
    manager.ace_cred_test_btn.setToolTip(
        "Ask Kaggle whether it accepts the credentials right now.\n\n"
        "This makes one real API call, so it fails for the same reasons a run "
        "would — a bad or expired token, no Internet, or a weights dataset that "
        "does not exist — but in seconds, with the reason shown, instead of "
        "after the audio has been staged and uploaded."
    )
    manager.ace_cred_setup_btn = QPushButton("🔑 Set up / change…")
    manager.ace_cred_setup_btn.setToolTip(
        "Enter the Kaggle username and API key. The dialog has its own "
        "\"remember on this device\" choice — unticked, the key is used for this "
        "session only and is never written anywhere."
    )
    manager.ace_cred_forget_btn = QPushButton("🗑 Forget stored key")
    manager.ace_cred_forget_btn.setToolTip(
        "Delete the stored Kaggle key from this device. It stays in use for the "
        "current session."
    )
    manager.ace_cred_reset_btn = QPushButton("↺ Ask again next run")
    manager.ace_cred_reset_btn.setToolTip(
        "Forget the once-per-backend decision about missing credentials, so the "
        "next caption run asks again instead of reusing the remembered fallback."
    )
    for _btn in (manager.ace_cred_setup_btn, manager.ace_cred_forget_btn,
                 manager.ace_cred_reset_btn):
        c_row.addWidget(_btn)
    c_row.addStretch()
    c_layout.addLayout(c_row)
    # manager.ace_cred_test_btn is deliberately NOT here: it is the first chip on
    # the strip, because "is Kaggle reachable?" is step zero of every run.

    # ------------------------------------------------------------------
    # Kaggle run (folders + dataset identity)
    # ------------------------------------------------------------------
    settings_layout.addWidget(_head("Kaggle run — folders &amp; dataset"))
    k_form = QFormLayout()
    settings_layout.addLayout(k_form)

    manager.caption_staging_edit = QLineEdit(
        (manager.config.get("caption_staging_dir") or "").strip() or default_staging_dir()
    )
    manager.caption_staging_edit.setToolTip(
        "The LOCAL folder that IS the uploaded Kaggle dataset.\n\n"
        "Everything in it is uploaded, so this is where you add songs (copy them "
        "in) and remove songs (select them below → Remove). Re-runs push a new "
        "VERSION of the same dataset instead of making a new one, so the dataset "
        "keeps its identity and its URL."
    )
    manager.caption_staging_browse_btn = QPushButton("Browse…")
    manager.caption_staging_browse_btn.setToolTip("Choose the staging folder.")
    k_form.addRow("Staging folder:",
                  _folder_row(manager.caption_staging_edit,
                              manager.caption_staging_browse_btn, ""))

    manager.caption_audio_dataset_edit = QLineEdit(
        manager.config.get("caption_audio_dataset", "")
    )
    manager.caption_audio_dataset_edit.setToolTip(
        "The private Kaggle dataset the audio is uploaded to, as 'user/slug'.\n\n"
        "Leave it empty on the first run: one is created and remembered here. "
        "After that, every run uploads a new version of THIS dataset, which is "
        "what lets you add or remove songs without losing the dataset."
    )
    k_form.addRow("Kaggle dataset:", manager.caption_audio_dataset_edit)

    manager.caption_model_dataset_edit = QLineEdit(
        manager.config.get("kaggle_model_dataset", "")
    )
    manager.caption_model_dataset_edit.setPlaceholderText("user/captioner-weights")
    manager.caption_model_dataset_edit.setToolTip(
        "The Kaggle dataset holding the captioner WEIGHTS, as 'user/slug'. It is "
        "attached to the kernel READ-ONLY — nothing is uploaded to it, and it is "
        "never modified.\n\n"
        "Default: michelmoalem9b/acestep-captioner-model (the community export). "
        "Keep it set to reuse that cached copy.\n\n"
        "Clear it and every session downloads ACE-Step/acestep-captioner from "
        "Hugging Face inside the kernel instead: ~16 GB per run, Internet must be "
        "ON, and a gated repo needs HF_TOKEN. The kernel also auto-detects weights "
        "from ANY dataset mounted at /kaggle/input that contains config.json plus "
        "safetensors, so a hand-attached dataset works without changing this."
    )
    k_form.addRow("Model weights dataset:", manager.caption_model_dataset_edit)

    manager.caption_output_edit = QLineEdit(
        (manager.config.get("caption_output_dir") or "").strip() or default_output_dir()
    )
    manager.caption_output_edit.setToolTip(
        "The LOCAL folder the captions are downloaded into (captions_out.json).\n\n"
        "Kaggle only persists /kaggle/working inside the kernel, so this is the "
        "path worth choosing: it is where the results actually land on this "
        "machine."
    )
    manager.caption_output_browse_btn = QPushButton("Browse…")
    manager.caption_output_browse_btn.setToolTip("Choose where captions are saved.")
    k_form.addRow("Output folder:",
                  _folder_row(manager.caption_output_edit,
                              manager.caption_output_browse_btn, ""))

    manager.caption_convert_check = QCheckBox(
        "Convert the dataset's audio to MP3 before uploading"
    )
    manager.caption_convert_check.setChecked(
        bool(manager.config.get("caption_convert_mp3", True))
    )
    manager.caption_convert_check.setToolTip(
        "Transcodes each track with ffmpeg into the staging folder. One codec "
        "instead of a mix of flac/wav/m4a, and a far smaller upload.\n\n"
        "The ORIGINAL file is staged at full quality when ffmpeg is missing or "
        "refuses a file — the captioner hears whatever is staged, so a track is "
        "never silently dropped."
    )
    k_form.addRow("", manager.caption_convert_check)

    manager.caption_bitrate_combo = QComboBox()
    manager.caption_bitrate_combo.setEditable(True)
    manager.caption_bitrate_combo.addItems(["128k", "192k", "256k", "320k"])
    manager.caption_bitrate_combo.setCurrentText(
        str(manager.config.get("caption_mp3_bitrate", "192k"))
    )
    manager.caption_bitrate_combo.setToolTip(
        "MP3 bitrate for the staged copies (used only when conversion is on)."
    )
    k_form.addRow("MP3 bitrate:", manager.caption_bitrate_combo)

    # ------------------------------------------------------------------
    # Prompt add-on + limits
    # ------------------------------------------------------------------
    settings_layout.addWidget(_head("Caption prompt &amp; limits"))
    p_form = QFormLayout()
    settings_layout.addLayout(p_form)

    manager.caption_addendum_edit = QTextEdit()
    manager.caption_addendum_edit.setPlainText(
        manager.config.get("caption_prompt_addendum", "")
    )
    manager.caption_addendum_edit.setMaximumHeight(110)
    manager.caption_addendum_edit.setPlaceholderText(
        "e.g. This is a 1970s live bootleg — mention tape hiss and the room. "
        "Always name the guitar amp."
    )
    manager.caption_addendum_edit.setToolTip(
        "APPENDED to the caption request, on top of the built-in ACE-Step 1.5XL "
        "schema in modules/caption_spec.py, which cannot be replaced.\n\n"
        "Use it for run-specific emphasis — era, room, artist, house style. An "
        "empty box changes nothing, and it is ignored by the instrument-only "
        "'Detect via Captioner' prompt, which is an explicit override."
    )
    p_form.addRow("Prompt add-on:", manager.caption_addendum_edit)

    manager.max_tokens_spin = _spin(
        64, 4096, int(manager.config.get("caption_max_tokens", 512))
    )
    manager.max_tokens_spin.setToolTip(
        "Maximum tokens the captioner may generate per track."
    )
    p_form.addRow("Max tokens:", manager.max_tokens_spin)

    manager.max_dur_spin = _spin(
        0, 3600, int(manager.config.get("caption_max_audio_duration", 120))
    )
    manager.max_dur_spin.setToolTip(
        "SECONDS PER PASS — not how much of the song you get.\n\n"
        "A pass is one model call, and it is capped by GPU memory: two T4s hold "
        "~120 s of audio comfortably, so this is the value that works. It is the "
        "first thing to lower again if runs start failing to load.\n\n"
        "0 = the whole file in one pass (no cutting at all — which is how a long "
        "song runs out of memory). The value actually used is printed in the "
        "kernel log."
    )
    p_form.addRow("Pass length (sec):", manager.max_dur_spin)

    manager.caption_whole_song_check = QCheckBox(
        "Caption the whole song (2+ passes per track)"
    )
    manager.caption_whole_song_check.setChecked(
        bool(manager.config.get("caption_whole_song", True))
    )
    manager.caption_whole_song_check.setToolTip(
        "ON: a song longer than one pass is captioned in several passes that "
        "together cover the WHOLE track, and their captions are merged into one — "
        "so the caption can describe how the song develops from beginning to end, "
        "which the schema demands.\n\n"
        "OFF: one pass only, so everything after the first “Pass length” seconds "
        "is never heard.\n\n"
        "COST: every pass is a separate GPU call, so this is roughly 2-3× the GPU "
        "time per track, and the run's wait budget grows with it."
    )
    p_form.addRow("", manager.caption_whole_song_check)

    manager.batch_size_spin = _spin(
        1, 64, int(manager.config.get("caption_batch_size", 1))
    )
    manager.batch_size_spin.setToolTip(
        "Tracks per forward pass on the captioning GPU. 1 is the safe value on "
        "T4s; raise it only if the runs have memory headroom."
    )
    p_form.addRow("Batch size:", manager.batch_size_spin)

    # ------------------------------------------------------------------
    # Staging: what the next run uploads (the folder's CONTENTS)
    # ------------------------------------------------------------------
    staging_grp = QGroupBox("Staged files — this folder IS the Kaggle dataset")
    s_layout = QVBoxLayout(staging_grp)

    s_hint = QLabel(
        "<b>➕ Stage</b> (on the strip) copies the ticked tracks into this folder; "
        "<b>Remove</b> deletes the ticked tracks again — the same ticks. The next "
        "run uploads the result as a new version of the dataset, so its identity "
        "survives. Removing a song here never touches the dataset on disk."
    )
    s_hint.setProperty("muted", True)
    s_hint.setWordWrap(True)
    s_layout.addWidget(s_hint)

    manager.staging_list = QListWidget()
    manager.staging_list.setSelectionMode(QListWidget.ExtendedSelection)
    manager.staging_list.setMaximumHeight(150)
    manager.staging_list.setToolTip(
        "Files currently staged for upload. Everything listed here is uploaded and "
        "captioned on the next run.\n\n"
        "Each row reads  staged file   ←   dataset track , because the upload uses "
        "the converted MP3 name while the dataset may still say .flac. A row marked "
        "“(not in this dataset)” is a leftover from an older dataset — removable "
        "here, and captioned by the kernel unless you remove it.\n\n"
        "Files that CANNOT be uploaded (0-byte corpses from an interrupted "
        "transcode, .part scratch, our own metadata file) are deliberately NOT "
        "listed — the line under this list names them, and 🧹 Clean deletes them."
    )
    s_layout.addWidget(manager.staging_list)

    s_row = QHBoxLayout()
    manager.staging_remove_btn = QPushButton("➖ Remove")
    manager.staging_remove_btn.setToolTip(
        "Delete the TICKED tracks — the same “Tracks ▾” ticks ➕ Stage uses — from "
        "the staging folder. This is how a song is removed from the uploaded "
        "dataset.\n\n"
        "A file clicked in the list is removed as well, which is the only way to get "
        "rid of a staged file that belongs to no track in this dataset."
    )
    manager.staging_clean_btn = QPushButton("🧹 Clean unusable")
    manager.staging_clean_btn.setToolTip(
        "Delete the files the upload would not take anyway: 0-byte files left by an "
        "interrupted transcode, and .part scratch files. They are invisible in the "
        "list above (it shows only what uploads), which is exactly why the folder "
        "and the list could silently disagree before."
    )
    manager.staging_refresh_btn = QPushButton("Refresh")
    manager.staging_refresh_btn.setToolTip("Re-read the staging folder from disk.")
    for _btn in (manager.staging_remove_btn, manager.staging_clean_btn,
                 manager.staging_refresh_btn):
        s_row.addWidget(_btn)
    s_row.addStretch()
    s_layout.addLayout(s_row)

    manager.staging_count_label = QLabel("Nothing staged yet.")
    manager.staging_count_label.setProperty("muted", True)
    s_layout.addWidget(manager.staging_count_label)
    layout.addWidget(staging_grp)

    # ------------------------------------------------------------------
    # The step widgets. Each is created here with its own tooltip and ADDED to
    # the strip at the end of this function, in workflow order — the strip is one
    # line, so its placement calls have to be in one place.
    # ------------------------------------------------------------------
    # THE track selector for this page. It is the ONLY way to choose which tracks
    # this page acts on: sending the user to the Dataset Studio table to set a ROW
    # selection — for work that happens here — was the wrong UI, and row selection
    # is not "a place to add tracks" in any case.
    manager.ace_track_picker = TrackPickerButton()
    manager.ace_track_picker.setToolTip(
        "Tick the tracks this page acts on. Staging, captioning, re-captioning and "
        "editing all use this list.\n\n"
        "“Select tracks missing captions” ticks only the ones still to do.\n\n"
        "Ticks are keyed by filename, so they survive table refreshes, dataset "
        "reloads and re-ordering."
    )
    manager.ace_tick_status = QLabel("")
    manager.ace_tick_status.setProperty("muted", True)
    manager.ace_tick_status.setWordWrap(True)

    # A review POLICY rather than a step, so it belongs in Settings.
    manager.caption_batch_review_check = QCheckBox(
        "Review the whole run in one diff table (recommended)"
    )
    manager.caption_batch_review_check.setChecked(
        bool(manager.config.get("caption_batch_review", True))
    )
    manager.caption_batch_review_check.setToolTip(
        "ON: every new caption is held as a proposal and one table shows all of "
        "them against what the tracks already have, so you choose once.\n\n"
        "OFF: the old behaviour — a dialog per track as the results arrive."
    )
    settings_layout.addWidget(manager.caption_batch_review_check)

    manager.staging_add_btn = QPushButton("➕ Stage")
    manager.staging_add_btn.setToolTip(
        "Copy (or transcode) the tracks ticked in “Tracks ▾” into the staging "
        "folder — that folder is what gets uploaded to Kaggle."
    )
    manager.caption_selected_btn = QPushButton("🚀 Caption")
    manager.caption_selected_btn.setToolTip(
        "Caption the tracks ticked in the Tracks dropdown above."
    )
    manager.caption_missing_btn = QPushButton("Caption missing")
    manager.caption_missing_btn.setToolTip(
        "Caption every track that has no caption yet — the whole dataset, not just "
        "the ticked ones."
    )
    manager.caption_all_btn = QPushButton("🔁 Caption all")
    manager.caption_all_btn.setToolTip(
        "Re-run the captioner over every track (asks for confirmation). An "
        "approved caption is never destroyed: the previous text is kept in "
        "caption_before_kaggle when you accept a replacement."
    )
    manager.caption_edit_btn = QPushButton("📝 Edit")
    manager.caption_edit_btn.setToolTip(
        "Edit the caption / lyrics of the ticked track. With several ticked, the "
        "diff table opens instead so you can decide each one."
    )
    manager.caption_recaption_bad_btn = QPushButton("♻ Re-caption")
    manager.caption_recaption_bad_btn.setToolTip(
        "Caption only the tracks that need it again: blank captions, tracks the "
        "kernel reported an error for, and tracks that got no result at all in "
        "the last run."
    )
    manager.caption_diff_btn = QPushButton("🔍 Review")
    manager.caption_diff_btn.setToolTip(
        "Open the diff table again: existing caption vs the last proposal, per "
        "track, with a decision for each."
    )
    manager.caption_import_btn = QPushButton("📥 Import")
    manager.caption_import_btn.setToolTip(
        "Load a captions_out.json you downloaded (or produced in a Kaggle "
        "notebook by hand) and diff it against the dataset without re-running "
        "the kernel."
    )
    # ------------------------------------------------------------------
    # THE STRIP, in workflow order. Adding the widgets HERE, rather than at each
    # creation site above, is what makes the order below the order of the WORKFLOW
    # instead of the order this file happens to be written in.
    # ------------------------------------------------------------------
    for _widget in (
        manager.ace_cred_test_btn,
        manager.ace_track_picker,
        manager.staging_add_btn,
        manager.caption_selected_btn,
        manager.caption_missing_btn,
        manager.caption_all_btn,
        manager.caption_edit_btn,
        manager.caption_diff_btn,
        manager.caption_recaption_bad_btn,
        manager.caption_import_btn,
    ):
        strip.addWidget(_widget)
    # The tick status is the only stretchable item: it absorbs the slack and
    # wraps, so the action buttons keep their natural width and nothing clips.
    strip.addWidget(manager.ace_tick_status)

    layout.addWidget(settings_grp)
    layout.addStretch()
    return inner
