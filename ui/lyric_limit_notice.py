"""The read-before-you-edit page for the ACE-Step lyric token limit.

ACE-Step cuts text at a token cap with ``truncation=True`` and says nothing
about it, so a dataset can be full of songs whose second half the model never
sang. This dialog is the app's one place that states the real cap, cites where
it is set in the ACE-Step source, and then gets out of the way.

Two exits, and they are deliberately different:

* **OK — let me trim my lyrics.** You may edit now, but the reminder comes back
  next time: it is a nudge, not a waiver.
* **I know what I'm doing — ...** records ``lyrics_limit_ack`` in settings.json,
  so the page never appears again for this user.

The live counter under the waveform (see
``dataset_manager.update_lyric_limit_label``) stays visible either way —
dismissing this page never hides the number.
"""
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)

import os

from modules import lyric_limits
from modules.lyric_limits import (
    CAPTION_TOKEN_LIMIT,
    LYRIC_INFER_TOKEN_LIMIT,
    LYRIC_TRAIN_TOKEN_LIMIT,
)

#: The full-sentence opt-out. Long on purpose: it is a joke with a real
#: consequence (it stops the reminder), not a hidden "don't show again".
BYPASS_TEXT = (
    "I know what I'm doing — I don't need to read what undocumented "
    "limitations you found in the source code"
)
TRIM_TEXT = "OK — let me trim my lyrics"


def _notice_html(limit, exact):
    """The page body. Kept as HTML so the caps can be a real table."""
    tokenizer_line = (
        "Counts are exact: the Qwen3-Embedding-0.6B <code>tokenizer.json</code> "
        "is installed, which is the same file ACE-Step's own "
        "<code>AutoTokenizer</code> loads."
        if exact
        else (
            "The exact Qwen3-Embedding-0.6B tokenizer is <b>not installed</b> "
            "yet, so the counter is a character estimate. Use the button below "
            "to fetch it (one time, about 11&nbsp;MB) and the numbers become "
            "exact."
        )
    )
    return f"""
<style>
  h2 {{ margin: 8px 0 6px 0; }}
  table {{ border-collapse: collapse; margin: 8px 0; }}
  th, td {{ border: 1px solid #666; padding: 3px 8px; text-align: left; }}
  code {{ background: rgba(128,128,128,0.18); padding: 0 3px; }}
</style>
<h2>ACE-Step truncates lyrics — silently</h2>
<p>Every piece of text ACE-Step 1.5 sends to the model is tokenised and then
<b>cut</b> at a fixed limit, with <code>truncation=True</code>. Past the limit
the rest is dropped: no error, no warning, nothing in the UI. The model simply
never sings those lines.</p>

<h2>The real limits (read from the ACE-Step 1.5 source)</h2>
<table>
  <tr><th>Stream</th><th>Cap</th><th>Set in</th></tr>
  <tr><td>Caption / DiT text prompt</td><td>{CAPTION_TOKEN_LIMIT} tokens</td>
      <td><code>conditioning_text.py</code></td></tr>
  <tr><td><b>Lyrics — training</b></td>
      <td><b>{LYRIC_TRAIN_TOKEN_LIMIT} tokens</b></td>
      <td><code>preprocess_lyrics.py</code></td></tr>
  <tr><td>Lyrics — inference</td><td>{LYRIC_INFER_TOKEN_LIMIT} tokens</td>
      <td><code>conditioning_text.py</code></td></tr>
</table>
<p>This app builds <b>training</b> datasets, so <b>{limit} tokens</b> is the
number that applies to the lyric files it writes.</p>

<h2>What that means for you</h2>
<ul>
  <li>A lyric file with more than <b>{limit} tokens</b> loses its tail during
      preprocessing. Whatever is past the cap never reaches the model.</li>
  <li>Structural tags (<code>[Verse 1]</code>, <code>[Chorus]</code>),
      punctuation and blank lines all consume tokens too — counting words is
      not enough.</li>
  <li>The live counter under the waveform tracks this as you type, and turns
      red on the tracks that are already over.</li>
</ul>
<p>{tokenizer_line}</p>
"""


class LyricLimitNotice(QDialog):
    """Modal page stating the lyric token cap, with a snarky opt-out."""

    def __init__(self, parent=None, tokenizer_path=None, limit=None):
        super().__init__(parent)
        self.setWindowTitle("Before you edit lyrics — ACE-Step token limit")
        self.resize(760, 600)
        self._tokenizer_path = tokenizer_path
        self._limit = limit or lyric_limits.DEFAULT_LYRIC_LIMIT
        #: Set True only by the snarky button, so the caller knows to persist
        #: the waiver rather than a mere acknowledgement.
        self.bypassed = False

        lay = QVBoxLayout(self)
        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(False)
        lay.addWidget(self.body, 1)

        row = QHBoxLayout()
        self.download_btn = QPushButton("Download the exact tokenizer (≈11 MB)")
        self.download_btn.setToolTip(
            "Fetch Qwen3-Embedding-0.6B/tokenizer.json from the ACE-Step "
            "Hugging Face repo so the counter is exact instead of estimated."
        )
        self.download_btn.clicked.connect(self._download)
        row.addWidget(self.download_btn)
        row.addStretch()

        self.trim_btn = QPushButton(TRIM_TEXT)
        self.trim_btn.setToolTip(
            "Continue editing. This reminder will appear again next time."
        )
        self.trim_btn.clicked.connect(self.accept)
        row.addWidget(self.trim_btn)

        self.bypass_btn = QPushButton(BYPASS_TEXT)
        self.bypass_btn.setToolTip(
            "Continue editing and never show this page again "
            "(saved to settings.json as 'lyrics_limit_ack')."
        )
        self.bypass_btn.clicked.connect(self._bypass)
        row.addWidget(self.bypass_btn)
        lay.addLayout(row)

        self._render()

    def _render(self):
        exact = lyric_limits.have_tokenizer(self._tokenizer_path)
        self.body.setHtml(_notice_html(self._limit, exact))
        self.download_btn.setVisible(not exact)

    def _download(self):
        try:
            lyric_limits.download_tokenizer(self._tokenizer_path)
        except Exception as e:  # noqa: BLE001 — report, never crash the dialog
            QMessageBox.warning(self, "Download failed", str(e))
            return
        self._render()

    def _bypass(self):
        self.bypassed = True
        self.accept()


def _headless():
    """True when there is no human who could read a modal page.

    Load-bearing, not a convenience: ``acknowledge_lyrics_limit`` is reached from
    a ``textChanged`` handler, so a modal ``exec()`` with no display would
    deadlock the caller (the test suite drives exactly that path). The test suite
    sets ``ACESTEP_SKIP_LYRICS_GATE`` in ``tests/conftest.py``.
    """
    if os.environ.get("ACESTEP_SKIP_LYRICS_GATE"):
        return True
    try:
        from PySide6.QtWidgets import QApplication

        if QApplication.instance() is None:
            return True
    except Exception:  # noqa: BLE001 — no Qt at all: definitely headless
        return True
    return False


def acknowledge_lyrics_limit(manager):
    """Gate a lyrics edit. True = go ahead, False = the user backed out.

    Shows the notice at most once per session when ``lyrics_limit_ack`` is not
    already set in settings.json. ``trim_btn`` allows the edit but does NOT
    persist, so the reminder returns; ``bypass_btn`` persists the waiver.
    """
    if getattr(manager, "_lyrics_limit_session_ok", False):
        return True
    config = getattr(manager, "config", None)
    if config is None:
        return True  # no config (tests / tooling): do not gate
    if config.get("lyrics_limit_ack"):
        manager._lyrics_limit_session_ok = True
        return True
    if _headless():
        return True

    dlg = LyricLimitNotice(
        manager,
        tokenizer_path=lyric_limits.tokenizer_path_from_config(config),
        limit=lyric_limits.lyric_limit_from_config(config),
    )
    allowed = dlg.exec() == QDialog.Accepted
    if allowed:
        manager._lyrics_limit_session_ok = True
        if dlg.bypassed:
            config["lyrics_limit_ack"] = True
            try:
                from modules.config_store import save_plain_keys

                save_plain_keys(config, ["lyrics_limit_ack"])
            except Exception:  # noqa: BLE001 — persistence is best-effort
                pass
    return allowed
