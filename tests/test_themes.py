"""Themes: readable contrast, valid overrides, no hard-coded widget colors."""
import os
import subprocess

import pytest

from ui.themes import BUILTIN_THEMES, DEFAULT_THEME, ROLES, build_stylesheet, resolve_palette

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
