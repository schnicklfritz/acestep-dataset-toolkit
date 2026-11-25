"""FlowLayout must survive being asked to lay out from inside its own pass.

Applying a stylesheet re-polishes every widget. The flow strip advertises
HeightForWidth, so Qt asks ``heightForWidth()`` again while the real
``setGeometry()`` pass is moving children; unguarded, the layout re-entered
itself until the GUI thread was wedged (observed on a stylesheet polish, which
walks every widget in the process). These tests pin the re-entrancy guard in
``ui/shell.py``.
"""
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect
from PySide6.QtWidgets import QPushButton, QWidget, QWidgetItem

from ui.shell import FlowLayout

ITEM_W, ITEM_H, SPACING = 100, 30, 6
COUNT = 4
ONE_ROW_W = COUNT * (ITEM_W + SPACING)      # 424: every item fits on one row


def _build(qapp, layout_cls=FlowLayout):
    host = QWidget()
    layout = layout_cls(host, spacing=SPACING)
    for i in range(COUNT):
        btn = QPushButton(f"button {i}", host)
        btn.setFixedSize(ITEM_W, ITEM_H)
        layout.addItem(QWidgetItem(btn))
    return host, layout


class _NestingHeightLayout(FlowLayout):
    """Re-enters ``heightForWidth()`` from inside ``_do_layout``, as Qt does."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent, spacing)
        self.nest = False
        self.nested_height = None

    def _do_layout(self, rect, test_only):
        if self.nest:
            self.nest = False
            self.nested_height = self.heightForWidth(rect.width())
        return super()._do_layout(rect, test_only)


class _NestingGeometryLayout(FlowLayout):
    """Re-enters ``setGeometry()`` from inside ``_do_layout`` (a child resize)."""

    def __init__(self, parent=None, spacing=6):
        super().__init__(parent, spacing)
        self.nest = False
        self.reentered = 0

    def _do_layout(self, rect, test_only):
        if self.nest:
            self.nest = False
            self.reentered += 1
            self.setGeometry(rect)
        return super()._do_layout(rect, test_only)


def test_height_for_width_is_stable(qapp):
    """Repeated calls at one width agree, so a polish pass can converge."""
    host, layout = _build(qapp)

    first = layout.heightForWidth(ONE_ROW_W)
    assert first == ITEM_H
    for _ in range(5):
        assert layout.heightForWidth(ONE_ROW_W) == first
    assert layout.hasHeightForWidth()


def test_height_for_width_counts_wrapped_rows(qapp):
    """Narrowing the strip reports every row, not just the first."""
    host, layout = _build(qapp)

    assert layout.heightForWidth(ONE_ROW_W) == ITEM_H
    assert layout.heightForWidth(ITEM_W) == COUNT * ITEM_H + (COUNT - 1) * SPACING


def test_nested_height_for_width_returns_last_computed_height(qapp):
    """A call from inside a pass must return, not recurse."""
    host, layout = _build(qapp, _NestingHeightLayout)
    warm = layout.heightForWidth(ONE_ROW_W)

    layout.nest = True
    layout.setGeometry(QRect(0, 0, ONE_ROW_W, warm))

    assert layout.nested_height == warm
    assert layout._laying_out is False


def test_nested_set_geometry_does_not_recurse(qapp):
    """A child resize that re-enters setGeometry leaves the pass in charge."""
    host, layout = _build(qapp, _NestingGeometryLayout)

    layout.nest = True
    layout.setGeometry(QRect(0, 0, ONE_ROW_W, ITEM_H))

    assert layout.reentered == 1                  # the guard, not a crash
    assert layout.geometry().width() == ONE_ROW_W
    assert layout.itemAt(0).geometry().topLeft().x() == 0
    assert layout._laying_out is False
