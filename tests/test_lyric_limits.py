"""The ACE-Step text token caps: the counter and the read-before-you-edit gate.

WHY THIS EXISTS
---------------
ACE-Step truncates every text input at a fixed token cap with
``truncation=True`` and reports nothing, so a lyric file can lose its tail
during preprocessing with no error anywhere. Two things in the app depend on
getting the numbers and the gate right, and both are pinned here:

* ``modules.lyric_limits`` — the caps and the token counting. The caps are
  literals in the ACE-Step source (caption 256 / lyrics-training 512 /
  lyrics-inference 2048); the count is exact when the Qwen3-Embedding-0.6B
  ``tokenizer.json`` is present and a stated estimate when it is not.
* ``ui.lyric_limit_notice`` — the modal page plus its two exits. The snarky
  bypass persists ``lyrics_limit_ack``; the trim button does not.

The gate is reached from ``lyrics_text.textChanged``, so it must never block a
headless run: ``conftest.py`` sets ``ACESTEP_SKIP_LYRICS_GATE``, and the tests
that exercise the gate delete it again.
"""
import os

import pytest

pytest.importorskip("PySide6")

from modules import lyric_limits as L  # noqa: E402

FAKE_PATH = "/nonexistent/Qwen3-Embedding-0.6B/tokenizer.json"


# --- counting ---------------------------------------------------------------
def test_measure_is_exact_with_the_real_tokenizer():
    if not L.have_tokenizer():
        pytest.skip("Qwen3 tokenizer.json not installed")
    m = L.measure("[Verse 1]\nhey jude dont make it bad\n")
    assert m.exact is True
    assert m.tokens > 0
    assert m.chars == len("[Verse 1]\nhey jude dont make it bad\n")


def test_measure_empty_text_is_zero():
    """The tokenizer reports 1 (its EOS); the UI must not show that as 'used'."""
    assert L.measure("").tokens == 0
    assert L.measure(None).tokens == 0


def test_measure_falls_back_to_an_estimate_and_says_so():
    m = L.measure("x" * 370, FAKE_PATH)
    assert m.exact is False
    assert m.tokens == 100  # 370 chars / 3.7 per token


def test_over_by_never_goes_negative():
    m = L.Measurement(tokens=10, chars=1, exact=True)
    assert m.over_by(512) == 0
    assert m.is_over(512) is False
    assert L.Measurement(tokens=520, chars=1, exact=True).over_by(512) == 8


# --- the caps are configurable, with the trainer's 512 as the default --------
@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, 512),
        ({"lyrics_max_tokens": 0}, 512),
        ({"lyrics_max_tokens": None}, 512),
        ({"lyrics_max_tokens": "junk"}, 512),
        ({"lyrics_max_tokens": 2048}, 2048),
        ({"lyrics_max_tokens": "2048"}, 2048),
        ({"lyrics_max_tokens": 300}, 300),
    ],
)
def test_lyric_limit_from_config(config, expected):
    assert L.lyric_limit_from_config(config) == expected


def test_tokenizer_path_from_config():
    assert L.tokenizer_path_from_config({}) == L.DEFAULT_TOKENIZER_PATH
    assert str(L.tokenizer_path_from_config({"lyrics_tokenizer_path": "/tmp/t.json"})) == "/tmp/t.json"


# --- the live counter line under the waveform -------------------------------
@pytest.fixture
def counter():
    """The real ``DatasetManager.update_lyric_limit_label`` on a stub host."""
    from PySide6.QtWidgets import QLabel, QPlainTextEdit

    import dataset_manager

    cls = next(
        obj for obj in vars(dataset_manager).values()
        if isinstance(obj, type) and hasattr(obj, "update_lyric_limit_label")
    )

    class _Stub:
        def __init__(self, config=None):
            self.lyrics_limit_label = QLabel()
            self.config = config if config is not None else {}
            self.lyrics_text = QPlainTextEdit()
            self.caption_text = QPlainTextEdit()

        def update(self):
            cls.update_lyric_limit_label(self)
            return self.lyrics_limit_label.text()

    return _Stub


def test_counter_reports_both_streams(counter, qapp):
    text = counter().update()
    assert "Lyrics 0 / 512 tokens" in text
    assert "Caption 0 / 256 tokens" in text


def test_counter_marks_the_over_limit_stream_red(counter, qapp):
    s = counter()
    s.lyrics_text.setPlainText("[Verse 1]\n" + "la la la la la la la la\n" * 200)
    text = s.update()
    assert "over by" in text
    assert "#d33" in text


def test_counter_labels_estimated_numbers(counter, qapp):
    text = counter({"lyrics_tokenizer_path": FAKE_PATH}).update()
    assert "≈" in text
    assert "estimated" in text


def test_counter_honours_the_configured_cap(counter, qapp):
    s = counter({"lyrics_max_tokens": 2048})
    s.lyrics_text.setPlainText("[Verse 1]\n" + "la la la la la la la la\n" * 200)
    assert " / 2048 tokens" in s.update()


# --- the read-before-you-edit gate ------------------------------------------
class _Mgr:
    """The only thing ``acknowledge_lyrics_limit`` needs from a manager."""

    def __init__(self, config):
        self.config = config


def _enable_gate(monkeypatch):
    """Turn the gate back on for one test (conftest disables it globally)."""
    monkeypatch.delenv("ACESTEP_SKIP_LYRICS_GATE", raising=False)


def _fake_dialog(monkeypatch, *, accepted, bypassed):
    import ui.lyric_limit_notice as notice

    class _Fake:
        def __init__(self, parent=None, tokenizer_path=None, limit=None):
            self.bypassed = bypassed
            self.limit = limit

        def exec(self):
            return 1 if accepted else 0

    monkeypatch.setattr(notice, "LyricLimitNotice", _Fake)


def test_gate_allows_editing_when_there_is_no_config():
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    assert acknowledge_lyrics_limit(object()) is True


def test_gate_allows_editing_when_already_acknowledged(qapp):
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    mgr = _Mgr({"lyrics_limit_ack": True})
    assert acknowledge_lyrics_limit(mgr) is True
    assert mgr._lyrics_limit_session_ok is True


def test_gate_never_blocks_a_headless_run(monkeypatch, qapp):
    """The suite's default state: no modal, and nothing written to config."""
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    monkeypatch.setenv("ACESTEP_SKIP_LYRICS_GATE", "1")
    mgr = _Mgr({})
    assert acknowledge_lyrics_limit(mgr) is True
    assert "lyrics_limit_ack" not in mgr.config


def test_trim_button_allows_the_edit_but_does_not_persist(monkeypatch, qapp):
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    _enable_gate(monkeypatch)
    _fake_dialog(monkeypatch, accepted=True, bypassed=False)
    saved = []
    monkeypatch.setattr(
        "modules.config_store.save_plain_keys", lambda cfg, keys: saved.append(keys)
    )
    mgr = _Mgr({})
    assert acknowledge_lyrics_limit(mgr) is True
    assert mgr.config.get("lyrics_limit_ack") is not True
    assert saved == []


def test_snarky_bypass_persists_the_acknowledgement(monkeypatch, qapp):
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    _enable_gate(monkeypatch)
    _fake_dialog(monkeypatch, accepted=True, bypassed=True)
    saved = []
    monkeypatch.setattr(
        "modules.config_store.save_plain_keys", lambda cfg, keys: saved.append(keys)
    )
    mgr = _Mgr({})
    assert acknowledge_lyrics_limit(mgr) is True
    assert mgr.config["lyrics_limit_ack"] is True
    assert saved == [["lyrics_limit_ack"]]


def test_declining_the_notice_blocks_the_edit(monkeypatch, qapp):
    from ui.lyric_limit_notice import acknowledge_lyrics_limit

    _enable_gate(monkeypatch)
    _fake_dialog(monkeypatch, accepted=False, bypassed=False)
    mgr = _Mgr({})
    assert acknowledge_lyrics_limit(mgr) is False
    assert "lyrics_limit_ack" not in mgr.config
    assert getattr(mgr, "_lyrics_limit_session_ok", False) is False


def test_notice_page_cites_the_real_caps(qapp):
    from ui.lyric_limit_notice import BYPASS_TEXT, LyricLimitNotice

    d = LyricLimitNotice(None, limit=512)
    body = d.body.toPlainText()
    assert "256" in body and "512" in body and "2048" in body
    assert d.bypass_btn.text() == BYPASS_TEXT
    assert "trim" in d.trim_btn.text().lower()
