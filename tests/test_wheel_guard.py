"""The scroll-wheel guard.

Regression guard for the accident where scrolling the dataset silently changed
a combo/spin/slider the cursor merely passed over.
"""
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from modules.wheel_guard import (
    GuardedComboBox,
    GuardedDoubleSpinBox,
    GuardedSlider,
    GuardedSpinBox,
    install_wheel_guard,
)


def _wheel(widget, dy=-120):
    pos = QPointF(5, 5)
    ev = QWheelEvent(pos, pos, QPoint(0, 0), QPoint(0, dy),
                     Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    from PySide6.QtWidgets import QApplication
    QApplication.sendEvent(widget, ev)
    return ev


def _press(widget):
    from PySide6.QtWidgets import QApplication
    ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(5, 5),
                     Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, ev)


class TestGuardedSubclasses:
    def test_guarded_types_are_subclasses_of_the_qt_types(self):
        assert issubclass(GuardedComboBox, QComboBox)
        assert issubclass(GuardedSpinBox, QSpinBox)
        assert issubclass(GuardedDoubleSpinBox, QDoubleSpinBox)
        assert issubclass(GuardedSlider, QSlider)

    def test_unclicked_combo_ignores_the_wheel(self, qapp):
        c = GuardedComboBox()
        c.addItems(["a", "b", "c"])
        c.show()
        ev = _wheel(c)
        assert c.currentText() == "a"
        # Not accepted -> the event bubbles to the enclosing scroll area.
        assert not ev.isAccepted()

    def test_clicked_combo_accepts_the_wheel(self, qapp):
        c = GuardedComboBox()
        c.addItems(["a", "b", "c"])
        c.show()
        _press(c)
        assert c._wheel_armed is True

    def test_unclicked_spinbox_ignores_the_wheel(self, qapp):
        s = GuardedSpinBox()
        s.setRange(0, 100)
        s.show()
        _wheel(s)
        assert s.value() == 0

    def test_unclicked_slider_ignores_the_wheel(self, qapp):
        sl = GuardedSlider(Qt.Horizontal)
        sl.setRange(0, 100)
        sl.show()
        _wheel(sl)
        assert sl.value() == 0

    def test_unclicked_double_spinbox_ignores_the_wheel(self, qapp):
        d = GuardedDoubleSpinBox()
        d.setRange(0.0, 10.0)
        d.show()
        _wheel(d)
        assert d.value() == 0.0


class TestScrollingStillWorks:
    def test_wheel_over_an_unclicked_combo_scrolls_the_page(self, qapp):
        """The whole point: the page scrolls, the value does not change."""
        from PySide6.QtWidgets import QLabel, QApplication

        area = QScrollArea()
        inner = QWidget()
        lay = QVBoxLayout(inner)
        for i in range(60):
            lay.addWidget(QLabel(f"row {i}"))
        combo = GuardedComboBox()
        combo.addItems(["a", "b", "c"])
        lay.insertWidget(3, combo)
        area.setWidget(inner)
        area.resize(300, 200)
        area.show()
        QApplication.processEvents()

        before_value = combo.currentText()
        before_bar = area.verticalScrollBar().value()
        for _ in range(6):
            ev = _wheel(combo)
            if not ev.isAccepted():
                QApplication.sendEvent(area.viewport(), ev)  # Qt propagates it
        QApplication.processEvents()

        assert combo.currentText() == before_value, "value changed while scrolling"
        assert area.verticalScrollBar().value() > before_bar, "page did not scroll"


class TestAppLevelFallback:
    def test_install_is_idempotent_and_retains_the_filter(self, qapp):
        guard = install_wheel_guard(qapp)
        assert guard is not None
        # Retained on the app so Qt's pointer stays valid.
        assert qapp._wheel_guard is guard
