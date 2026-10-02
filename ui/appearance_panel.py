"""Settings > Appearance: theme picker, per-color editor, font, text size, zoom.

Every change applies immediately to the whole app and is saved with
``save_plain_keys`` (settings.json only; secrets are never touched).

The text-size controls here and the one on the lyrics field's actions row are
copies of ONE setting each (``ui_font_size`` / ``lyrics_font_size``), kept in
step by ``sync_font_size_controls``. Neither is a second source of truth.
"""
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QColorDialog, QFontComboBox, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QPushButton, QWidget,
)

from modules.wheel_guard import GuardedComboBox as QComboBox
from modules.wheel_guard import GuardedSlider as QSlider
from modules.wheel_guard import GuardedSpinBox as QSpinBox
from ui.themes import (
    FONT_SIZE_MAX,
    LYRICS_FONT_SIZE_MAX,
    ROLES,
    clamp_font_size,
    resolve_font_sizes,
    resolve_palette,
    theme_names,
)

APPEARANCE_KEYS = ("theme_name", "theme_overrides", "ui_font_family",
                   "ui_font_size", "lyrics_font_size", "ui_zoom")


def build_font_size_spin(tooltip, size=0, maximum=FONT_SIZE_MAX):
    """A px size control whose minimum is ``Auto`` (0 = inherit).

    One builder for the global text size, the lyrics override, and the inline
    copy on the lyrics field's actions row, so the copies cannot drift in range
    or in what "Auto" means. ``maximum`` is parameterised because the lyrics
    surfaces have a higher ceiling than the rest of the app
    (``LYRICS_FONT_SIZE_MAX``, see ui/themes.py) — passing the app-wide
    ``FONT_SIZE_MAX`` for a lyrics spinner would make the control refuse a value
    ``clamp_font_size`` still accepts, and the refusal would be silent (the spin
    would just snap back).
    """
    spin = QSpinBox()
    spin.setRange(0, maximum)
    spin.setSingleStep(1)
    spin.setSpecialValueText("Auto")
    spin.setSuffix(" px")
    spin.setValue(clamp_font_size(size, maximum=maximum))
    spin.setToolTip(tooltip)
    return spin


def sync_font_size_controls(manager):
    """Re-read the font-size spinners from config so no two copies disagree.

    The lyrics field's actions row has its own spinner bound to the same key as
    the Settings row; when one changes, the other is pushed back in line here
    rather than keeping a stale number for the same setting.
    """
    base, lyrics = resolve_font_sizes(manager.config)
    explicit = clamp_font_size(manager.config.get("ui_font_size"))
    for name, value in (("font_size_spin", base if explicit else 0),
                        ("lyrics_font_spin", lyrics or 0),
                        ("lyrics_font_spin_inline", lyrics or 0)):
        spin = getattr(manager, name, None)
        if spin is None or spin.value() == value:
            continue
        spin.blockSignals(True)
        spin.setValue(value)
        spin.blockSignals(False)


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
    manager.font_picker.currentFontChanged.connect(lambda f: set_appearance(manager, "ui_font_family", f.family()))
    form.addRow("Font:", manager.font_picker)

    manager.font_size_spin = build_font_size_spin(
        "Text size for the whole app, in pixels. Auto follows the built-in size "
        "(13 px). Anything you set here is scaled by Zoom.",
        manager.config.get("ui_font_size"),
    )
    manager.font_size_spin.valueChanged.connect(lambda v: set_appearance(manager, "ui_font_size", v))
    form.addRow("Text size:", manager.font_size_spin)

    manager.lyrics_font_spin = build_font_size_spin(
        "Text size for the lyrics fields only — the inline Formatted Lyrics box "
        "and the Lyrics Studio window. Auto means they match the rest of the app, "
        "which is what you want unless you are reading lyrics from a distance.",
        manager.config.get("lyrics_font_size"),
        maximum=LYRICS_FONT_SIZE_MAX,
    )
    manager.lyrics_font_spin.valueChanged.connect(lambda v: set_appearance(manager, "lyrics_font_size", v))
    form.addRow("Lyrics size:", manager.lyrics_font_spin)

    zoom_row = QHBoxLayout()
    manager.zoom_slider = QSlider(Qt.Horizontal)
    manager.zoom_slider.setRange(75, 175)
    pct = round(float(manager.config.get("ui_zoom") or 1.0) * 100)
    manager.zoom_slider.setValue(pct)
    manager.zoom_label = QLabel(f"{pct}%")
    manager.zoom_slider.valueChanged.connect(lambda v: (manager.zoom_label.setText(f"{v}%"),
                                                        set_appearance(manager, "ui_zoom", v / 100.0)))
    zoom_row.addWidget(manager.zoom_slider, 1)
    zoom_row.addWidget(manager.zoom_label)
    form.addRow("Zoom:", zoom_row)

    manager.theme_combo.currentTextChanged.connect(lambda n: set_appearance(manager, "theme_name", n))
    refresh_swatches(manager)
    return grp


def refresh_swatches(manager):
    _name, palette = resolve_palette(manager.config)
    for role, btn in manager.theme_swatches.items():
        # A swatch shows a literal color by definition; it is the one place a
        # widget-level stylesheet is correct.
        btn.setStyleSheet(f"background: {palette[role]}; border: 1px solid {palette['border']}; border-radius: 4px;")


def set_appearance(manager, key, value):
    manager.config[key] = value
    # Scoped: a slider step must not re-polish the whole process. An Appearance
    # change never creates a top-level window, so re-scoping to the one window
    # is enough and is what stops the GUI thread stalling for seconds per step.
    manager.apply_custom_theme(scoped=True)
    # A size change may have come from the lyrics field's own header spinner;
    # mirror it into the Settings copy (and vice versa) so the two never show
    # different numbers for the same key.
    sync_font_size_controls(manager)
    _save(manager)


def _pick_color(manager, role):
    name, palette = resolve_palette(manager.config)
    col = QColorDialog.getColor(QColor(palette[role]), manager, f"{ROLES[role]} — {name}")
    if not col.isValid():
        return
    overrides = {k: dict(v) for k, v in (manager.config.get("theme_overrides") or {}).items()}
    overrides.setdefault(name, {})[role] = col.name()
    set_appearance(manager, "theme_overrides", overrides)


def _reset_colors(manager):
    name, _ = resolve_palette(manager.config)
    overrides = {k: dict(v) for k, v in (manager.config.get("theme_overrides") or {}).items()}
    overrides.pop(name, None)
    set_appearance(manager, "theme_overrides", overrides)


def _save(manager):
    from modules.config_store import save_plain_keys

    save_plain_keys(manager.config, APPEARANCE_KEYS)
