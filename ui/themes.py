"""Color themes: two built-ins plus per-color overrides stored in settings.json.

A theme is a flat palette of named ROLES. The stylesheet is generated from the
palette, so a user override of one role (say ``accent``) restyles every widget
that uses it. Widgets never hard-code colors; they opt into a look with a
dynamic property instead:

    label.setProperty("muted", True)        # secondary text
    label.setProperty("tone", "warning")    # warning / danger / positive text
    button.setProperty("role", "primary")   # the one main action in an area
    button.setProperty("role", "danger")    # destructive or "armed" state
    frame.setProperty("health", "warn")     # left-border status strip

After changing a property at runtime call ``repolish(widget)``.

Config keys (see config.DEFAULT_CONFIG): ``theme_name``, ``theme_overrides``
(``{theme_name: {role: "#rrggbb"}}``), ``ui_font_family``, ``ui_zoom``.
"""
import re

# role -> label shown in the color editor. Order is the editor's order.
ROLES = {
    "window": "Window background",
    "surface": "Panels",
    "surface_alt": "Raised / hover",
    "input": "Input fields",
    "border": "Borders",
    "text": "Text",
    "text_muted": "Secondary text",
    "accent": "Accent",
    "accent_text": "Text on accent",
    "selection": "Selection",
    "positive": "Success",
    "warning": "Warning",
    "danger": "Danger",
}

BUILTIN_THEMES = {
    # Polished dark: neutral graphite, one cool accent, low-contrast borders so
    # structure comes from spacing rather than boxes.
    "Studio Dark": {
        "window": "#15171c",
        "surface": "#1c1f26",
        "surface_alt": "#262a33",
        "input": "#111317",
        "border": "#2c313b",
        "text": "#e4e7ec",
        "text_muted": "#8d96a5",
        "accent": "#3d6fe0",
        "accent_text": "#ffffff",
        "selection": "#2b3d66",
        "positive": "#4fbf87",
        "warning": "#e3a33b",
        "danger": "#e5595b",
    },
    # Light, flat, teal accent -- the look of the layout mockup.
    "Paper Teal": {
        "window": "#f4f2ed",
        "surface": "#fbfaf7",
        "surface_alt": "#ebe8e1",
        "input": "#ffffff",
        "border": "#d9d4c9",
        "text": "#20242a",
        "text_muted": "#6a6f78",
        "accent": "#0f766e",
        "accent_text": "#ffffff",
        "selection": "#c9e6e2",
        "positive": "#23724a",
        "warning": "#8a5a0c",
        "danger": "#c2410c",
    },
}

DEFAULT_THEME = "Studio Dark"
_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def theme_names():
    return list(BUILTIN_THEMES)


def resolve_palette(config):
    """Built-in palette for ``config['theme_name']`` with that theme's overrides.

    Invalid override values are ignored rather than breaking the stylesheet.
    """
    name = config.get("theme_name") or DEFAULT_THEME
    if name not in BUILTIN_THEMES:
        name = DEFAULT_THEME
    palette = dict(BUILTIN_THEMES[name])
    overrides = (config.get("theme_overrides") or {}).get(name) or {}
    for role, value in overrides.items():
        if role in palette and isinstance(value, str) and _HEX.match(value):
            palette[role] = value.lower()
    return name, palette


def _mix(a, b, t):
    """Blend two #rrggbb colors; t=0 -> a, t=1 -> b."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def is_dark(palette):
    r, g, b = (int(palette["window"][i:i + 2], 16) for i in (1, 3, 5))
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) < 128


_ICONS = {
    "check": '<path d="M3.5 8.5l3 3 6-7" fill="none" stroke="{c}" stroke-width="2" '
             'stroke-linecap="round" stroke-linejoin="round"/>',
    "dot": '<circle cx="8" cy="8" r="3.2" fill="{c}"/>',
    "chevron": '<path d="M4.5 6.5l3.5 3.5 3.5-3.5" fill="none" stroke="{c}" stroke-width="1.6" '
               'stroke-linecap="round" stroke-linejoin="round"/>',
}


def _icon(name, color):
    """Write a tiny themed SVG once and return a stylesheet-safe path.

    Qt stylesheets can only draw check marks and arrows from image files, and
    the color has to follow the palette, so the files are generated.
    """
    import os
    import tempfile

    d = os.path.join(tempfile.gettempdir(), "acestep_toolkit_icons")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{name}_{color.lstrip('#')}.svg")
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                     + _ICONS[name].format(c=color) + "</svg>")
    return path.replace("\\", "/")


def build_stylesheet(p, font_family="", zoom=1.0):
    """Qt stylesheet for palette ``p``."""
    check, dot, chevron = (_icon("check", p["accent_text"]), _icon("dot", p["accent_text"]),
                           _icon("chevron", p["text_muted"]))
    zoom = max(0.75, min(1.75, float(zoom or 1.0)))
    fs = round(13 * zoom)
    small = max(9, fs - 2)
    pad_v, pad_h = round(5 * zoom), round(12 * zoom)
    radius = round(6 * zoom)
    font = f"font-family: '{font_family}';" if font_family else ""
    accent_hover = _mix(p["accent"], p["text"], 0.15)
    accent_press = _mix(p["accent"], p["window"], 0.2)
    btn_hover = _mix(p["surface_alt"], p["text"], 0.06)
    return f"""
* {{ {font} font-size: {fs}px; }}
QWidget {{ background: {p['window']}; color: {p['text']}; }}
QMainWindow::separator {{ background: {p['window']}; width: 6px; height: 6px; }}
QMainWindow::separator:hover {{ background: {p['accent']}; }}

QDockWidget {{ color: {p['text_muted']}; font-weight: 600; }}
QDockWidget::title {{ background: {p['window']}; padding: 6px 10px; text-align: left; }}
QDockWidget > QWidget {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: {radius}px; }}

QToolBar {{ background: {p['window']}; border: none; spacing: 6px; padding: 6px 8px; }}
QToolBar QLabel {{ background: transparent; color: {p['text_muted']}; }}
QStatusBar {{ background: {p['window']}; color: {p['text_muted']}; }}
QMenuBar {{ background: {p['window']}; }}
QMenuBar::item:selected {{ background: {p['surface_alt']}; border-radius: 4px; }}
QMenu {{ background: {p['surface']}; border: 1px solid {p['border']}; padding: 4px; }}
QMenu::item {{ padding: 6px 18px; border-radius: 4px; }}
QMenu::item:selected {{ background: {p['selection']}; }}
QMenu::separator {{ height: 1px; background: {p['border']}; margin: 4px 6px; }}
QToolTip {{ background: {p['surface_alt']}; color: {p['text']}; border: 1px solid {p['border']}; padding: 6px; }}

QGroupBox {{ background: {p['surface']}; border: 1px solid {p['border']}; border-radius: {radius}px;
            margin-top: 1.4em; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; color: {p['text_muted']};
                   font-weight: 600; background: transparent; }}
QGroupBox::indicator {{ width: 14px; height: 14px; }}
QGroupBox:flat {{ background: transparent; border: none; margin-top: 0; padding: 2px 0; }}
QGroupBox[collapsed="true"] {{ background: transparent; border: none; margin-top: 1.4em; padding: 0; }}
QLabel {{ background: transparent; }}
QLabel[muted="true"] {{ color: {p['text_muted']}; }}
QLabel[small="true"] {{ font-size: {small}px; }}
QLabel[tone="warning"] {{ color: {p['warning']}; }}
QLabel[tone="danger"] {{ color: {p['danger']}; }}
QLabel[tone="positive"] {{ color: {p['positive']}; }}
QFrame[health], QLabel[health] {{ background: {p['surface_alt']}; border-radius: 4px; padding: 6px 8px;
                                   border-left: 3px solid {p['border']}; }}
QFrame[health="warn"], QLabel[health="warn"] {{ border-left-color: {p['warning']}; }}
QFrame[health="ok"], QLabel[health="ok"] {{ border-left-color: {p['positive']}; }}
QFrame[health="bad"], QLabel[health="bad"] {{ border-left-color: {p['danger']}; }}

QPushButton, QToolButton {{ background: {p['surface_alt']}; color: {p['text']}; border: 1px solid {p['border']};
                           border-radius: {radius}px; padding: {pad_v}px {pad_h}px; }}
QPushButton:hover, QToolButton:hover {{ background: {btn_hover}; }}
QPushButton:pressed, QToolButton:pressed {{ background: {p['surface']}; }}
QPushButton:checked, QToolButton:checked {{ background: {p['selection']}; border-color: {p['accent']}; }}
QPushButton:disabled, QToolButton:disabled {{ color: {p['text_muted']}; background: {p['surface']}; }}
QPushButton[role="primary"] {{ background: {p['accent']}; color: {p['accent_text']}; border-color: {p['accent']};
                              font-weight: 600; }}
QPushButton[role="primary"]:hover {{ background: {accent_hover}; }}
QPushButton[role="primary"]:pressed {{ background: {accent_press}; }}
QPushButton[role="danger"], QPushButton[role="danger"]:checked {{ background: {p['danger']};
                              color: {p['accent_text']}; border-color: {p['danger']}; font-weight: 600; }}
QToolButton::menu-indicator {{ width: 0; }}
QToolBar QToolButton {{ background: transparent; border-color: transparent; }}
QToolBar QToolButton:hover {{ background: {p['surface_alt']}; border-color: {p['border']}; }}

QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QFontComboBox {{
    background: {p['input']}; color: {p['text']}; border: 1px solid {p['border']}; border-radius: {radius}px;
    padding: 4px 8px; selection-background-color: {p['selection']}; selection-color: {p['text']}; }}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {p['accent']}; }}
QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {p['text_muted']}; }}
QComboBox::drop-down {{ border: none; width: 22px; }}
QComboBox::down-arrow {{ image: url("{chevron}"); width: 14px; height: 14px; }}
QComboBox QAbstractItemView {{ background: {p['surface']}; border: 1px solid {p['border']};
                              selection-background-color: {p['selection']}; outline: none; }}
QCheckBox, QRadioButton {{ background: transparent; spacing: 6px; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; border: 1px solid {p['border']};
                                                background: {p['input']}; }}
QCheckBox::indicator {{ border-radius: 3px; }}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; image: url("{check}"); }}
QRadioButton::indicator:checked {{ background: {p['accent']}; border-color: {p['accent']}; image: url("{dot}"); }}
QGroupBox::indicator:checked {{ background: {p['accent']}; border: 1px solid {p['accent']}; border-radius: 3px; image: url("{check}"); }}
QGroupBox::indicator:unchecked {{ background: {p['input']}; border: 1px solid {p['border']}; border-radius: 3px; }}

QTableWidget, QTableView, QListWidget, QListView, QTreeWidget, QTreeView {{
    background: {p['surface']}; alternate-background-color: {_mix(p['surface'], p['surface_alt'], 0.5)};
    border: 1px solid {p['border']}; border-radius: {radius}px; gridline-color: {p['border']};
    selection-background-color: {p['selection']}; selection-color: {p['text']}; outline: none; }}
QTableView::item, QListView::item, QTreeView::item {{ padding: 3px 4px; }}
QHeaderView::section {{ background: {p['surface']}; color: {p['text_muted']}; border: none;
                       border-bottom: 1px solid {p['border']}; padding: 6px 6px; font-weight: 600; }}
QTableCornerButton::section {{ background: {p['surface']}; border: none; }}

QTabWidget::pane {{ border: 1px solid {p['border']}; border-radius: {radius}px; background: {p['surface']}; top: -1px; }}
QTabBar::tab {{ background: transparent; color: {p['text_muted']}; padding: 7px 14px; border: none;
               border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {p['text']}; border-bottom-color: {p['accent']}; }}
QTabBar::tab:hover {{ color: {p['text']}; }}
QToolBox::tab {{ background: {p['surface_alt']}; color: {p['text']}; border: 1px solid {p['border']};
                border-radius: {radius}px; padding: 7px 10px; font-weight: 600; }}
QToolBox::tab:selected {{ background: {p['selection']}; border-color: {p['accent']}; }}

QScrollArea {{ background: transparent; border: none; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle {{ background: {p['surface_alt']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::handle:hover {{ background: {_mix(p['surface_alt'], p['text'], 0.2)}; }}
QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ background: none; height: 0; width: 0; }}
QSplitter::handle {{ background: {p['window']}; }}
QSplitter::handle:hover {{ background: {p['accent']}; }}
QProgressBar {{ background: {p['surface_alt']}; border: none; border-radius: 4px; height: 8px;
               color: transparent; max-height: 8px; }}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 4px; }}
QSlider::groove:horizontal {{ height: 4px; background: {p['surface_alt']}; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: {p['accent']}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {p['text']}; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }}
"""


def apply_theme(app, config):
    """Apply the configured theme to the whole QApplication (dialogs included)."""
    from PySide6.QtGui import QColor, QPalette

    name, p = resolve_palette(config)
    app.setStyleSheet(build_stylesheet(p, config.get("ui_font_family") or "", config.get("ui_zoom") or 1.0))
    # Native pieces the stylesheet does not reach (color dialog, some popups).
    pal = QPalette()
    for role, key in [
        (QPalette.Window, "window"), (QPalette.Base, "input"), (QPalette.AlternateBase, "surface"),
        (QPalette.Button, "surface_alt"), (QPalette.Text, "text"), (QPalette.WindowText, "text"),
        (QPalette.ButtonText, "text"), (QPalette.Highlight, "selection"), (QPalette.HighlightedText, "text"),
        (QPalette.ToolTipBase, "surface_alt"), (QPalette.ToolTipText, "text"),
        (QPalette.PlaceholderText, "text_muted"), (QPalette.Link, "accent"),
    ]:
        pal.setColor(role, QColor(p[key]))
    app.setPalette(pal)
    return name, p


def repolish(widget):
    """Re-evaluate the stylesheet after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
