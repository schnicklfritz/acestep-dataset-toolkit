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
(``{theme_name: {role: "#rrggbb"}}``), ``ui_font_family``, ``ui_font_size``,
``lyrics_font_size``, ``ui_zoom``.

Text size has two layers, and they are deliberately not the same knob:
``ui_font_size`` is the base size every widget inherits, while
``lyrics_font_size`` overrides it for the lyrics surfaces only (the inline
field and the Lyrics Studio window). ``ui_zoom`` is a third thing again — it
magnifies the whole look, padding and corner radius included.

``0`` means "Auto": for ``ui_font_size`` that is ``DEFAULT_FONT_SIZE``, and for
``lyrics_font_size`` it means "same as the rest of the app".
"""
import re

# The base text size in px when ``ui_font_size`` is Auto, and the range both
# size controls offer. Before this existed the base was the literal 13 buried
# in build_stylesheet().
DEFAULT_FONT_SIZE = 13
FONT_SIZE_MIN, FONT_SIZE_MAX = 9, 32

# Lyrics get their own ceiling above the app-wide one. Lyrics are read from a
# distance, in a field the user can expand, and 32px is not enough for that:
# measured, the collapsed field (a fixed-height box) showed only 4 lines at the
# old cap. The raise is a SEPARATE constant, not a bigger FONT_SIZE_MAX, so the
# rest of the app keeps its tested 9-32 range. It matters that the ceiling is
# real rather than cosmetic: ``clamp_font_size`` maps any out-of-range value to
# 0 (Auto), so without this a hand-edited ``lyrics_font_size`` of 48 would not
# merely be refused by the spinner — it would silently snap back to Auto/13px.
LYRICS_FONT_SIZE_MAX = 72

# Widgets that should follow the lyrics size mark themselves with this dynamic
# property, and build_stylesheet() emits one rule for them. Why a stylesheet
# rule and not QWidget.setFont(): the app stylesheet's ``* { font-size }``
# overrides any font a widget sets on itself the next time the theme is
# applied (measured — a setFont(26px) field snapped back to the base size on a
# theme switch), whereas a property rule outranks ``*`` and survives the
# re-apply. See ui/lyrics_studio.py and DatasetManager.init_ui for the users.
FONT_REGION_PROPERTY = "fontRegion"
FONT_REGION_LYRICS = "lyrics"

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


def clamp_font_size(value, minimum=FONT_SIZE_MIN, maximum=FONT_SIZE_MAX):
    """Coerce a stored ``ui_font_size`` / ``lyrics_font_size`` to a usable px int.

    Settings files are hand-editable and get written by older versions of the
    app, so a value here can be a string, ``None``, or nonsense. Anything that
    is not an int in range becomes ``0`` — "Auto", i.e. "follow the default" —
    rather than raising, because a bad number in settings.json must not stop
    the window from opening.
    """
    try:
        size = int(float(value))
    except (TypeError, ValueError):
        return 0
    return size if minimum <= size <= maximum else 0


def resolve_font_sizes(config):
    """Return ``(base_px, lyrics_px)`` for a config dict.

    ``base_px`` is what the whole app uses; ``lyrics_px`` is what the lyrics
    surfaces use, and is ``None`` when they are set to Auto — meaning "emit no
    override rule at all", so they inherit ``base_px`` exactly.
    """
    base = clamp_font_size((config or {}).get("ui_font_size")) or DEFAULT_FONT_SIZE
    lyrics = clamp_font_size((config or {}).get("lyrics_font_size"),
                             maximum=LYRICS_FONT_SIZE_MAX)
    return base, (lyrics or None)


def build_stylesheet(p, font_family="", zoom=1.0, font_size=None, lyrics_font_size=None):
    """Qt stylesheet for palette ``p``.

    ``font_size`` is the base text size in px (``None`` keeps
    ``DEFAULT_FONT_SIZE``, the historical hard-coded 13). ``lyrics_font_size``
    is the override for widgets carrying the ``FONT_REGION_LYRICS`` property;
    ``None`` emits no rule, which is the Auto case.
    """
    check, dot, chevron = (_icon("check", p["accent_text"]), _icon("dot", p["accent_text"]),
                           _icon("chevron", p["text_muted"]))
    zoom = max(0.75, min(1.75, float(zoom or 1.0)))
    fs = round((clamp_font_size(font_size) or DEFAULT_FONT_SIZE) * zoom)
    # The lyrics override is scaled by zoom as well, so enlarging the lyrics
    # relative to the app survives a zoom change instead of inverting.
    lyrics = clamp_font_size(lyrics_font_size, maximum=LYRICS_FONT_SIZE_MAX)
    region_rule = (f'\n[{FONT_REGION_PROPERTY}="{FONT_REGION_LYRICS}"] '
                   f'{{ font-size: {round(lyrics * zoom)}px; }}') if lyrics else ""
    small = max(9, fs - 2)
    pad_v, pad_h = round(5 * zoom), round(12 * zoom)
    radius = round(6 * zoom)
    font = f"font-family: '{font_family}';" if font_family else ""
    accent_hover = _mix(p["accent"], p["text"], 0.15)
    accent_press = _mix(p["accent"], p["window"], 0.2)
    btn_hover = _mix(p["surface_alt"], p["text"], 0.06)
    return f"""
* {{ {font} font-size: {fs}px; }}{region_rule}
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


def _qpalette(p):
    """QPalette for palette ``p`` — the native pieces the stylesheet misses."""
    from PySide6.QtGui import QColor, QPalette

    pal = QPalette()
    for role, key in [
        (QPalette.Window, "window"), (QPalette.Base, "input"), (QPalette.AlternateBase, "surface"),
        (QPalette.Button, "surface_alt"), (QPalette.Text, "text"), (QPalette.WindowText, "text"),
        (QPalette.ButtonText, "text"), (QPalette.Highlight, "selection"), (QPalette.HighlightedText, "text"),
        (QPalette.ToolTipBase, "surface_alt"), (QPalette.ToolTipText, "text"),
        (QPalette.PlaceholderText, "text_muted"), (QPalette.Link, "accent"),
    ]:
        pal.setColor(role, QColor(p[key]))
    return pal


def _stylesheet(config):
    """The stylesheet for ``config`` — one place, so scope cannot change the look."""
    _name, p = resolve_palette(config)
    font_size, lyrics_size = resolve_font_sizes(config)
    return build_stylesheet(
        p, config.get("ui_font_family") or "", config.get("ui_zoom") or 1.0,
        font_size=font_size, lyrics_font_size=lyrics_size,
    )


def apply_theme(app, config):
    """Apply the configured theme to the whole QApplication (dialogs included).

    Setting the stylesheet on the QApplication re-polishes every widget in the
    process. That is correct at startup and for a one-off change, but it is what
    makes a live Appearance/zoom control freeze the GUI thread for seconds (see
    ``apply_theme_to``). Prefer this only when new top-level windows may appear.
    """
    _name, p = resolve_palette(config)
    app.setStyleSheet(_stylesheet(config))
    app.setPalette(_qpalette(p))
    return _name, p


def apply_theme_to(widget, config, app=None):
    """Apply the theme to ``widget`` and its Qt children only.

    The stylesheet is set on the WINDOW, not on the QApplication, so a slider
    step re-polishes one window instead of the whole process. Measured on this
    app: ~1 s scoped vs ~6 s app-wide per step, and the difference is entirely
    re-polishing widgets the change cannot affect (menus, other docks, dialogs).

    Windows and dialogs opened with this widget as their Qt parent inherit the
    sheet, because a stylesheet applies to a widget's whole QObject subtree; the
    application palette is still set so the few detached popups (tooltips) stay
    readable. Falls back to app-wide when ``widget`` is ``None``.
    """
    _name, p = resolve_palette(config)
    if widget is not None:
        widget.setStyleSheet(_stylesheet(config))
    elif app is not None:
        app.setStyleSheet(_stylesheet(config))
    if app is not None:
        app.setPalette(_qpalette(p))
    return _name, p


def repolish(widget):
    """Re-evaluate the stylesheet after a dynamic property change."""
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
