"""Main-window shell: track list + editor in the center, docked side panels.

Layout (the approved mockup):

    [ toolbar: Open / Save / Add / Export · dataset settings · Settings · Panels ]
    [ track table | track editor ]   [ Tools dock: Track / Caption / Lyrics / ... ]
                                     [ Assistant dock                           ]
    [ status bar: status text · progress ]

The pages are the same QWidgets the old top-level tabs used, re-parented into
docks, so every ``manager.<attr>`` reference and handler still resolves.

RETAINMENT: PySide6 frees a C++ object once its last Python reference goes
away, and freeing a parent frees its children. Every container built here is
therefore stored on the manager.
"""
import base64

from PySide6.QtCore import QPoint, QRect, QSize, Qt
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDialog, QDockWidget, QFrame, QGroupBox, QLayout, QScrollArea, QSizePolicy,
    QTabWidget, QVBoxLayout, QWidget, QWidgetItem,
)

DOCK_STATE_VERSION = 1


class FlowLayout(QLayout):
    """Left-to-right layout that wraps onto new lines (the Qt 'flow' example).

    Tool buttons live in a narrow dock; a QHBoxLayout would force the dock to
    the width of its longest row.
    """

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent)
        self._items = []
        self._spacing = spacing
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        return size + QSize(m.left() + m.right(), m.top() + m.bottom())

    def _do_layout(self, rect, test_only):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for item in self._items:
            if item.widget() is not None and not item.widget().isVisible() and not test_only:
                pass
            hint = item.sizeHint()
            next_x = x + hint.width() + self._spacing
            if next_x - self._spacing > r.right() and line_h > 0:
                x, y = r.x(), y + line_h + self._spacing
                next_x = x + hint.width() + self._spacing
                line_h = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def flow_group(title, box_layout):
    """Move every widget of a QHBoxLayout into a titled, wrapping group."""
    grp = QGroupBox(title)
    flow = FlowLayout(grp)
    flow.setContentsMargins(8, 12, 8, 8)
    while box_layout.count():
        item = box_layout.takeAt(0)
        w = item.widget()
        if w is not None:
            flow.addItem(QWidgetItem(w))
            w.setParent(grp)
        elif item.layout() is not None:
            # a nested row: flatten it
            sub = item.layout()
            while sub.count():
                s = sub.takeAt(0)
                if s.widget() is not None:
                    flow.addItem(QWidgetItem(s.widget()))
                    s.widget().setParent(grp)
        # spacers / stretches are dropped: a flow layout packs left
    grp.setSizePolicy(height_for_width_policy())
    return grp


def height_for_width_policy():
    """Size policy that makes the parent layout ask a FlowLayout how tall it
    must be at the current width (otherwise it gets one row and clips)."""
    sp = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    sp.setHeightForWidth(True)
    return sp


def make_collapsible(group):
    """A checkable QGroupBox only DISABLES its children when unchecked; hide
    them instead, so "collapsed" takes no space."""
    from ui.themes import repolish

    def apply(checked):
        for child in group.findChildren(QWidget, options=Qt.FindDirectChildrenOnly):
            child.setVisible(checked)
        group.setProperty("collapsed", not checked)
        repolish(group)
    group.toggled.connect(apply)
    apply(group.isChecked())


def _find_parent(layout, target):
    """(layout that directly contains ``target``, index) or (None, None)."""
    for i in range(layout.count()):
        sub = layout.itemAt(i).layout()
        if sub is target:
            return layout, i
        if sub is not None:
            found = _find_parent(sub, target)
            if found[0] is not None:
                return found
    return None, None


def fit_to_narrow_panel(page):
    """Let a page built for a full-width tab shrink to a dock's width.

    Labels wrap, form rows put the field under a long label, and nested
    scroll areas stop scrolling sideways. Nothing is hidden or removed.
    """
    from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel

    for lab in page.findChildren(QLabel):
        if lab.text() and not lab.pixmap():
            lab.setWordWrap(True)
            lab.setSizePolicy(QSizePolicy.Preferred, lab.sizePolicy().verticalPolicy())
    for form in page.findChildren(QFormLayout):
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    for area in page.findChildren(QScrollArea):
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    # Rows of 3+ plain buttons wrap instead of forcing the width.
    from PySide6.QtWidgets import QPushButton
    for row in page.findChildren(QHBoxLayout):
        widgets = [row.itemAt(i).widget() for i in range(row.count())]
        buttons = [w for w in widgets if isinstance(w, QPushButton)]
        if len(buttons) >= 3 and len(buttons) == len([w for w in widgets if w is not None]):
            host = row.parentWidget()
            if host is None or host.layout() is None:
                continue
            parent_layout, idx = _find_parent(host.layout(), row)
            if parent_layout is None:
                continue
            holder = QWidget(host)
            flow = FlowLayout(holder)
            for b in buttons:
                row.removeWidget(b)
                flow.addWidget(b)
            parent_layout.insertWidget(idx, holder)
            parent_layout.removeItem(row)


def _ensure_scroll(manager, page):
    """Give each page exactly ONE vertical scroll bar.

    A page with sub-tabs gets a scroll area per sub-tab that lacks one; a page
    that already scrolls itself is left alone; anything else is wrapped.
    Nested scroll bars eat width and clip text.
    """
    inner_tabs = [t for t in page.findChildren(QTabWidget)]
    if inner_tabs:
        for tabs in inner_tabs:
            for j in range(tabs.count()):
                sub = tabs.widget(j)
                if not sub.findChildren(QScrollArea) and not isinstance(sub, QScrollArea):
                    text, tip = tabs.tabText(j), tabs.tabToolTip(j)
                    tabs.removeTab(j)
                    area = _scroll(sub)
                    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                    manager._shell_scrolls.append(area)
                    tabs.insertTab(j, area, text)
                    tabs.setTabToolTip(j, tip)
        return page
    if page.findChildren(QScrollArea):
        return page
    area = _scroll(page)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    manager._shell_scrolls.append(area)
    return area


def _scroll(page):
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setWidget(page)
    return area


def _column(*items, stretch_last=True, margins=(0, 0, 0, 0), spacing=8):
    w = QWidget()
    v = QVBoxLayout(w)
    v.setContentsMargins(*margins)
    v.setSpacing(spacing)
    for n, it in enumerate(items):
        if it is None:
            continue
        s = 1 if (stretch_last and n == len(items) - 1) else 0
        if isinstance(it, QLayout):
            v.addLayout(it, s)
        else:
            v.addWidget(it, s)
    return w


def install_shell(manager, parts, tool_pages, assistant_page, settings_page):
    """Build the mockup layout from the pieces init_ui created.

    ``parts``: header, dataset_box, bulk, view_row, player, waveform, filter,
    table, inspector (see init_ui). ``tool_pages``: ordered [(label, QWidget)].
    """
    from PySide6.QtWidgets import QMenu, QSplitter, QToolBar, QToolButton

    manager.setDockNestingEnabled(True)

    # ---- toolbar: file actions + dataset-wide settings ----------------------
    tb = QToolBar("Main", manager)
    tb.setObjectName("toolbar_main")
    tb.setMovable(False)
    parts["header"].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    tb.addWidget(parts["header"])
    manager.addToolBar(Qt.TopToolBarArea, tb)
    manager.addToolBarBreak(Qt.TopToolBarArea)
    tb2 = QToolBar("Dataset", manager)
    tb2.setObjectName("toolbar_dataset")
    tb2.setMovable(False)
    box = parts["dataset_box"]
    box.setTitle("")
    box.setFlat(True)
    box.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    tb2.addWidget(box)
    manager.addToolBar(Qt.TopToolBarArea, tb2)
    manager._shell_toolbars = [tb, tb2]

    # ---- center: track list | editor ---------------------------------------
    tracks_col = _column(parts["filter"], parts["view_row"], parts["table"])
    for grp in parts["bulk"].findChildren(QGroupBox):
        if grp.isCheckable():
            make_collapsible(grp)
    editor_col = _column(parts["player"], parts["waveform"], parts["bulk"], parts["inspector"])
    manager.center_split = QSplitter(Qt.Horizontal)
    manager.center_split.setChildrenCollapsible(False)
    manager.center_split.addWidget(tracks_col)
    manager.center_split.addWidget(editor_col)
    manager.center_split.setStretchFactor(0, 6)
    manager.center_split.setStretchFactor(1, 5)
    manager.center_split.setSizes([520, 430])
    manager._shell_center = _column(manager.center_split, margins=(10, 4, 4, 4))
    manager.setCentralWidget(manager._shell_center)
    manager._shell_cols = [tracks_col, editor_col]
    install_column_menu(manager, parts["table"])

    # ---- Tools dock ----------------------------------------------------------
    # ``manager.tabs`` keeps its old name: _on_tab_changed and the
    # jump-to-track code still address pages by index through it.
    manager.tabs = QTabWidget()
    manager.tabs.setDocumentMode(True)
    manager._shell_scrolls = []
    for label, page in tool_pages:
        fit_to_narrow_panel(page)
        area = _ensure_scroll(manager, page)
        manager.tabs.addTab(area, label)
        manager._tab_pages.append(page)
    manager.tabs.currentChanged.connect(manager._on_tab_changed)

    manager.tools_dock = QDockWidget("Tools", manager)
    manager.tools_dock.setObjectName("dock_tools")
    manager.tools_dock.setWidget(manager.tabs)
    manager.tools_dock.setMinimumWidth(360)
    manager.addDockWidget(Qt.RightDockWidgetArea, manager.tools_dock)

    manager.assistant_dock = QDockWidget("Assistant", manager)
    manager.assistant_dock.setObjectName("dock_assistant")
    manager.assistant_dock.setWidget(assistant_page)
    manager._tab_pages.append(assistant_page)
    manager.addDockWidget(Qt.RightDockWidgetArea, manager.assistant_dock)
    # Stationary: no dragging it out to float (which is how it got lost off the
    # edge of the screen), and it may only live in the right-hand column. It can
    # still be tucked away — see install_collapsible_titlebar — but only to its
    # title bar, so the dock itself never disappears from the column.
    # PySide6 6.x dropped ``setFloatable`` (obsolete since Qt 6.2), so the flag
    # is cleared on the feature set directly.
    features = manager.assistant_dock.features()
    features &= ~QDockWidget.DockWidgetFeature.DockWidgetFloatable
    manager.assistant_dock.setFeatures(features)
    manager.assistant_dock.setAllowedAreas(Qt.RightDockWidgetArea)
    install_collapsible_titlebar(manager, manager.assistant_dock, "Assistant")
    manager.splitDockWidget(manager.tools_dock, manager.assistant_dock, Qt.Vertical)
    _default_sizes(manager)

    # ---- Settings: a window, not a tab (visited rarely) ---------------------
    manager.settings_dialog = QDialog(manager)
    manager.settings_dialog.setWindowTitle("Settings")
    manager.settings_dialog.resize(820, 760)
    lay = QVBoxLayout(manager.settings_dialog)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(settings_page)
    manager._tab_pages.append(settings_page)

    manager.settings_action = QAction("⚙ Settings", manager)
    manager.settings_action.setShortcut(QKeySequence("Ctrl+,"))
    manager.settings_action.triggered.connect(lambda: open_settings(manager))
    panels = QMenu(manager)
    for a in panels_menu_actions(manager):
        panels.addAction(a)
    panels.addSeparator()
    panels.addAction("Reset layout", lambda: reset_layout(manager))
    manager.panels_menu = panels
    head = parts["header"].layout()
    for text, action, menu in (("Panels", None, panels), (None, manager.settings_action, None)):
        btn = QToolButton()
        if action is not None:
            btn.setDefaultAction(action)
        else:
            btn.setText(text)
            btn.setMenu(menu)
            btn.setPopupMode(QToolButton.InstantPopup)
        head.addWidget(btn)
        manager._shell_toolbars.append(btn)

    # ---- status bar ------------------------------------------------------------
    sb = manager.statusBar()
    sb.setSizeGripEnabled(False)
    sb.addWidget(manager.status_label, 1)
    manager.progress_bar.setFixedWidth(260)
    sb.addPermanentWidget(manager.progress_bar)

    restore_layout(manager)


# Table columns hidden by default in the narrow track list; every column is
# still one right-click on the header away, and all fields are in the editor.
# Re-exported from ui.track_table, which owns the column policy: the list used
# to be defined here AND re-set ``Stretch`` on column 0 behind
# ``init_ui``'s ``ResizeToContents`` loop, so the two disagreed silently.
from ui.track_table import (  # noqa: E402,F401  (re-exported for callers/tests)
    DEFAULT_HIDDEN_COLUMNS,
    apply_hidden_columns,
)


def install_column_menu(manager, table):
    """Add the show/hide-column menu to ``table``'s header.

    Widths, resize modes and default visibility belong to
    ``ui.track_table.install_table_policy`` (called from ``init_ui``); this only
    layers the header menu on top, so there is exactly one place that decides
    what a column looks like.
    """
    from PySide6.QtWidgets import QMenu

    header = table.horizontalHeader()
    # Re-apply from config (not a hard-coded Stretch): a saved visibility
    # choice must win even if init_ui already set defaults.
    apply_hidden_columns(manager, table)
    names = [table.horizontalHeaderItem(i).text() for i in range(table.columnCount())]

    def menu(pos):
        m = QMenu(table)
        for i, n in enumerate(names):
            if i == 0:
                continue
            a = m.addAction(n)
            a.setCheckable(True)
            a.setChecked(not table.isColumnHidden(i))
            a.toggled.connect(lambda on, col=i: _toggle_column(manager, table, names, col, on))
        m.exec(header.mapToGlobal(pos))

    header.setContextMenuPolicy(Qt.CustomContextMenu)
    header.customContextMenuRequested.connect(menu)
    header.setToolTip(
        "Right-click to show or hide columns. Drag a column edge to resize it."
    )


def _toggle_column(manager, table, names, col, on):
    from modules.config_store import save_plain_keys

    table.setColumnHidden(col, not on)
    manager.config["ui_hidden_columns"] = [n for i, n in enumerate(names) if table.isColumnHidden(i)]
    save_plain_keys(manager.config, ["ui_hidden_columns"])


def install_collapsible_titlebar(manager, dock, title):
    """Replace ``dock``'s title bar with one that has a − / ▾ collapse button.

    A stock QDockWidget title bar offers only float and close — neither is
    "tuck this panel out of the way for a minute", and closing it is how the
    whole right-hand column ends up with nothing to click. This title bar keeps
    the panel where it is (the dock is ``setFloatable(False)`` with a
    right-column-only area) and collapses it to just the bar: it remembers the
    height, caps it to the bar's height, and restores it on the next click. The
    panel is minimized, never moved and never closed.
    """
    from PySide6.QtWidgets import QHBoxLayout, QLabel, QToolButton

    # Qt's ``QWIDGETSIZE_MAX`` macro; PySide6 does not re-export it, and it is
    # what ``setMaximumHeight`` needs to mean "no ceiling".
    unconstrained = 16777215

    bar = QWidget(dock)
    row = QHBoxLayout(bar)
    row.setContentsMargins(6, 2, 4, 2)
    row.setSpacing(4)
    label = QLabel(title)
    label.setProperty("muted", True)
    row.addWidget(label)
    row.addStretch()
    toggle = QToolButton(bar)
    toggle.setAutoRaise(True)
    row.addWidget(toggle)

    state = {"collapsed": False, "height": 0}

    def on_toggle():
        if state["collapsed"]:
            dock.setMaximumHeight(unconstrained)
            toggle.setText("−")
            toggle.setToolTip("Hide the panel (it stays in this column).")
            if state["height"]:
                manager.resizeDocks([dock], [state["height"]], Qt.Vertical)
        else:
            state["height"] = max(dock.height(), bar.sizeHint().height())
            dock.setMaximumHeight(bar.sizeHint().height())
            toggle.setText("▾")
            toggle.setToolTip("Show the panel.")
        state["collapsed"] = not state["collapsed"]

    toggle.clicked.connect(on_toggle)
    toggle.setText("−")
    toggle.setToolTip("Hide the panel (it stays in this column).")
    dock.setTitleBarWidget(bar)
    # Retained on the dock: Qt does not take ownership of a title-bar widget's
    # children, and a garbage-collected bar would blank the title.
    dock._collapse_bar = bar
    dock._collapse_button = toggle
    return toggle


def _default_sizes(manager):
    manager.resizeDocks([manager.tools_dock, manager.assistant_dock], [560, 320], Qt.Vertical)
    manager.resizeDocks([manager.tools_dock], [500], Qt.Horizontal)


def open_settings(manager=None):
    if manager is None:
        return
    manager.settings_dialog.show()
    manager.settings_dialog.raise_()
    manager.settings_dialog.activateWindow()


def panels_menu_actions(manager):
    return [manager.tools_dock.toggleViewAction(), manager.assistant_dock.toggleViewAction()]


def save_layout(manager):
    from modules.config_store import save_plain_keys

    state = bytes(manager.saveState(DOCK_STATE_VERSION))
    manager.config["ui_dock_state"] = base64.b64encode(state).decode("ascii")
    save_plain_keys(manager.config, ["ui_dock_state"])


def restore_layout(manager):
    raw = manager.config.get("ui_dock_state") or ""
    if not raw:
        return False
    try:
        restored = bool(manager.restoreState(base64.b64decode(raw), DOCK_STATE_VERSION))
    except Exception:  # noqa: BLE001 -- a bad saved state falls back to the default
        return False
    if restored:
        _repair_hidden_docks(manager)
    return restored


def _repair_hidden_docks(manager):
    """Never leave BOTH right-column docks hidden after a restore.

    A saved state can hide a dock on purpose — hiding only the Assistant is a
    legitimate, tested preference and is left alone. But if a state (or a
    half-collapsed column) leaves BOTH Tools and Assistant hidden, the entire
    right-hand column is gone and there is nothing left to click: the only way
    back is the Panels menu or Reset layout, which a user who never hid anything
    does not know to look for. This is the one case worth repairing instead of
    honouring, so both are shown and given their default sizes.
    """
    docks = (manager.tools_dock, manager.assistant_dock)
    if any(dock.isVisible() for dock in docks):
        return
    for dock in docks:
        dock.show()
    _default_sizes(manager)


def reset_layout(manager):
    from modules.config_store import save_plain_keys

    for dock in (manager.tools_dock, manager.assistant_dock):
        dock.setFloating(False)
        dock.show()
        manager.addDockWidget(Qt.RightDockWidgetArea, dock)
    manager.splitDockWidget(manager.tools_dock, manager.assistant_dock, Qt.Vertical)
    _default_sizes(manager)
    manager.config["ui_dock_state"] = ""
    save_plain_keys(manager.config, ["ui_dock_state"])
