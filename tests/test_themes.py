"""Themes: readable contrast, valid overrides, no hard-coded widget colors."""
import os
import subprocess

import pytest

from ui.themes import (
    BUILTIN_THEMES,
    DEFAULT_FONT_SIZE,
    DEFAULT_THEME,
    FONT_REGION_LYRICS,
    FONT_REGION_PROPERTY,
    ROLES,
    build_stylesheet,
    clamp_font_size,
    resolve_font_sizes,
    resolve_palette,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _lum(h):
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def contrast(a, b):
    hi, lo = sorted([_lum(a), _lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# WCAG 2.x: 4.5:1 for body text, 3:1 for UI components and large text.
TEXT_PAIRS = [("text", "window"), ("text", "surface"), ("text", "input"),
              ("text_muted", "surface"), ("text_muted", "window"),
              ("accent_text", "accent"), ("text", "selection"),
              ("warning", "surface"), ("danger", "surface"), ("positive", "surface")]


@pytest.mark.parametrize("name", list(BUILTIN_THEMES))
@pytest.mark.parametrize("fg,bg", TEXT_PAIRS)
def test_text_contrast(name, fg, bg):
    p = BUILTIN_THEMES[name]
    assert contrast(p[fg], p[bg]) >= 4.5, (name, fg, bg, round(contrast(p[fg], p[bg]), 2))


@pytest.mark.parametrize("name", list(BUILTIN_THEMES))
def test_accent_is_visible_as_a_ui_element(name):
    p = BUILTIN_THEMES[name]
    assert contrast(p["accent"], p["surface"]) >= 3.0


def test_every_theme_defines_every_role():
    for name, p in BUILTIN_THEMES.items():
        assert set(p) == set(ROLES), name


def test_default_is_studio_dark_and_unknown_falls_back():
    assert DEFAULT_THEME == "Studio Dark"
    assert resolve_palette({})[0] == "Studio Dark"
    assert resolve_palette({"theme_name": "gone"})[0] == "Studio Dark"


def test_overrides_apply_per_theme_and_bad_values_are_ignored():
    cfg = {"theme_name": "Paper Teal",
           "theme_overrides": {"Paper Teal": {"accent": "#ABCDEF", "text": "red", "bogus": "#000000"},
                               "Studio Dark": {"accent": "#111111"}}}
    _name, p = resolve_palette(cfg)
    assert p["accent"] == "#abcdef"
    assert p["text"] == BUILTIN_THEMES["Paper Teal"]["text"]
    assert "bogus" not in p


def test_stylesheet_uses_the_palette():
    p = dict(BUILTIN_THEMES["Studio Dark"], accent="#123456")
    assert "#123456" in build_stylesheet(p)


def test_no_widget_hard_codes_a_color():
    """Colors come from the theme; widgets opt in with properties
    (muted / tone / role / health). ui/appearance_panel.py is the exception:
    a color swatch shows a literal color by definition."""
    out = subprocess.run(
        ["git", "grep", "-n", "-E", r"setStyleSheet\(.*#[0-9A-Fa-f]{3}", "--", "*.py",
         ":!ui/themes.py", ":!ui/appearance_panel.py", ":!tests/*"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert out.stdout.strip() == "", out.stdout


# ---------------------------------------------------------------------------
# Text size: one global base, one lyrics-only override
# ---------------------------------------------------------------------------

def _sheet(**kw):
    return build_stylesheet(BUILTIN_THEMES["Studio Dark"], "", **kw)


def test_the_base_size_is_still_13_by_default():
    """``build_stylesheet`` hard-coded 13 for every caller before the setting
    existed. Adding the parameter must not move that default, or every user's
    app silently changes size on upgrade."""
    assert " font-size: 13px;" in _sheet()
    assert resolve_font_sizes({})[0] == DEFAULT_FONT_SIZE == 13


def test_a_scan_of_a_missing_size_setting_falls_back_to_the_default():
    """settings.json is written by older versions and is hand-editable, so the
    key may be absent (or junk) and the window must still open."""
    assert resolve_font_sizes({"ui_font_size": None})[0] == 13
    assert resolve_font_sizes({}) == (13, None)


def test_the_lyrics_override_is_absent_when_it_is_auto():
    """Auto means "inherit", which is only true if no rule is emitted — an
    emitted rule equal to the base would break the moment the base changes."""
    assert FONT_REGION_PROPERTY not in _sheet(font_size=20)
    assert resolve_font_sizes({"ui_font_size": 20})[1] is None


def test_the_lyrics_rule_is_emitted_and_carries_the_size():
    sheet = _sheet(font_size=20, lyrics_font_size=26)
    assert f'[{FONT_REGION_PROPERTY}="{FONT_REGION_LYRICS}"] {{ font-size: 26px; }}' in sheet


@pytest.mark.parametrize("raw,expected", [
    (None, 0), ("", 0), ("abc", 0), (0, 0),
    (7, 0),                 # below FONT_SIZE_MIN -> Auto, not a 7px app
    (999, 0),               # above FONT_SIZE_MAX -> Auto
    ("18", 18), (18.7, 18), # settings.json round-trips through JSON
])
def test_a_nonsense_stored_size_degrades_to_auto(raw, expected):
    assert clamp_font_size(raw) == expected


def test_zoom_scales_both_sizes():
    """Zoom is a whole-look magnifier; if it scaled only the base, a larger
    lyrics override would eventually be overtaken by the base."""
    sheet = _sheet(font_size=20, lyrics_font_size=26, zoom=1.5)
    assert " font-size: 30px;" in sheet
    assert 'font-size: 39px;' in sheet


def test_apply_theme_passes_both_sizes_through(qapp):
    from ui.themes import apply_theme

    apply_theme(qapp, {"theme_name": "Studio Dark", "ui_font_size": 20,
                       "lyrics_font_size": 26})
    sheet = qapp.styleSheet()
    assert " font-size: 20px;" in sheet
    assert f'[{FONT_REGION_PROPERTY}="{FONT_REGION_LYRICS}"]' in sheet


# ---------------------------------------------------------------------------
# Scoped application: a live slider step must not re-polish the whole process
# ---------------------------------------------------------------------------
# Setting the stylesheet on the QApplication re-polishes EVERY widget, which is
# what made one Appearance/zoom step stall the GUI thread for seconds. The
# scoped path sets it on the window instead, so one window is re-polished.

def test_apply_theme_to_scopes_the_stylesheet_to_the_widget(qapp):
    from PySide6.QtWidgets import QWidget

    from ui.themes import apply_theme, apply_theme_to

    apply_theme(qapp, {"theme_name": "Studio Dark"})
    app_sheet = qapp.styleSheet()
    w = QWidget()
    try:
        apply_theme_to(w, {"theme_name": "Studio Dark", "ui_font_size": 21})
        assert " font-size: 21px;" in w.styleSheet()
        assert w.styleSheet() != app_sheet
        assert qapp.styleSheet() == app_sheet          # application sheet untouched
    finally:
        w.deleteLater()


def test_apply_theme_to_falls_back_to_the_app_without_a_widget(qapp):
    from ui.themes import apply_theme_to

    apply_theme_to(None, {"theme_name": "Studio Dark", "ui_font_size": 22}, qapp)
    assert " font-size: 22px;" in qapp.styleSheet()
