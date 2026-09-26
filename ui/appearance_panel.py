"""Settings > Appearance: theme picker, per-color editor, font, zoom.

Every change applies immediately to the whole app and is saved with
``save_plain_keys`` (settings.json only; secrets are never touched).
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog, QFontComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QWidget,
)

from modules.wheel_guard import GuardedComboBox as QComboBox
from modules.wheel_guard import GuardedSlider as QSlider
from ui.themes import ROLES, resolve_palette, theme_names

APPEARANCE_KEYS = ("theme_name", "theme_overrides", "ui_font_family", "ui_zoom")


def build_appearance_group(manager):
    grp = QGroupBox("Appearance")
    form = QFormLayout(grp)

    manager.theme_combo = QComboBox()
    manager.theme_combo.addItems(theme_names())
    manager.theme_combo.setCurrentText(resolve_palette(manager.config)[0])
    manager.theme_combo.setToolTip("Studio Dark is the default. Paper Teal is a light, flat alternative.")
    form.addRow("Theme:", manager.theme_combo)

    swatch_box = QWidget()
    grid = QGridLayout(swatch_box)
    grid.setContentsMargins(0, 0, 0, 0)
    grid.setHorizontalSpacing(10)
    manager.theme_swatches = {}
    for i, (role, label) in enumerate(ROLES.items()):
        btn = QPushButton()
        btn.setFixedSize(34, 22)
        btn.setToolTip(f"{label} — click to change")
        btn.clicked.connect(lambda _=False, r=role: _pick_color(manager, r))
        manager.theme_swatches[role] = btn
        cell = QHBoxLayout()
        cell.addWidget(btn)
        name = QLabel(label)
        name.setProperty("muted", True)
        cell.addWidget(name, 1)
        grid.addLayout(cell, i // 2, i % 2)
    form.addRow("Colors:", swatch_box)

    reset = QPushButton("Reset this theme's colors")
    reset.setToolTip("Drop your color changes for the selected theme only.")
    reset.clicked.connect(lambda: _reset_colors(manager))
    form.addRow("", reset)

    manager.font_picker = QFontComboBox()
    fam = manager.config.get("ui_font_family") or ""
    if fam:
        manager.font_picker.setCurrentFont(QFont(fam))
    manager.font_picker.currentFontChanged.connect(lambda f: _set(manager, "ui_font_family", f.family()))
    form.addRow("Font:", manager.font_picker)

    zoom_row = QHBoxLayout()
    manager.zoom_slider = QSlider(Qt.Horizontal)
    manager.zoom_slider.setRange(75, 175)
    pct = round(float(manager.config.get("ui_zoom") or 1.0) * 100)
    manager.zoom_slider.setValue(pct)
    manager.zoom_label = QLabel(f"{pct}%")
    manager.zoom_slider.valueChanged.connect(lambda v: (manager.zoom_label.setText(f"{v}%"),
                                                        _set(manager, "ui_zoom", v / 100.0)))
    zoom_row.addWidget(manager.zoom_slider, 1)
    zoom_row.addWidget(manager.zoom_label)
    form.addRow("Zoom:", zoom_row)

    manager.theme_combo.currentTextChanged.connect(lambda n: _set(manager, "theme_name", n))
    refresh_swatches(manager)
    return grp


def refresh_swatches(manager):
    _name, palette = resolve_palette(manager.config)
    for role, btn in manager.theme_swatches.items():
        # A swatch shows a literal color by definition; it is the one place a
        # widget-level stylesheet is correct.
        btn.setStyleSheet(f"background: {palette[role]}; border: 1px solid {palette['border']}; border-radius: 4px;")


def _set(manager, key, value):
    manager.config[key] = value
    manager.apply_custom_theme()
    _save(manager)


def _pick_color(manager, role):
    name, palette = resolve_palette(manager.config)
    col = QColorDialog.getColor(QColor(palette[role]), manager, f"{ROLES[role]} — {name}")
    if not col.isValid():
        return
    overrides = {k: dict(v) for k, v in (manager.config.get("theme_overrides") or {}).items()}
    overrides.setdefault(name, {})[role] = col.name()
    _set(manager, "theme_overrides", overrides)


def _reset_colors(manager):
    name, _ = resolve_palette(manager.config)
    overrides = {k: dict(v) for k, v in (manager.config.get("theme_overrides") or {}).items()}
    overrides.pop(name, None)
    _set(manager, "theme_overrides", overrides)


def _save(manager):
    from modules.config_store import save_plain_keys

    save_plain_keys(manager.config, APPEARANCE_KEYS)
