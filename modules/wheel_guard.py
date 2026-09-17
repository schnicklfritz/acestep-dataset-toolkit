"""Scroll-wheel guard for value widgets.

Problem this solves: scrolling a long list (the dataset table, a settings pane)
moves the cursor across ``QComboBox`` / ``QSpinBox`` / ``QSlider`` widgets on
the way past. Qt delivers the wheel event to whatever is *under the cursor*, so
the combo silently cycles to its next value. The user never clicked anything and
usually doesn't notice — the value just changed.

WHY "FOCUS" IS NOT THE UNLOCK
-----------------------------
The obvious gate is ``hasFocus()``, but focus is granted for reasons other than
deliberate interaction: the first widget in a pane can receive it on show, and
tabbing or clicking a neighbouring row can hand it over. A widget can therefore
hold focus while the user is only scrolling — and the accident survives.

The reliable signal for "the user is working with this control" is an actual
**mouse press on it**, so the guards below arm themselves on press and disarm
when focus leaves. That is the only state that requires intentional interaction.

- never pressed -> wheel is ignored by the widget and bubbles to the enclosing
                scroll area, so the page scrolls. Value is unchanged.
- pressed     -> normal behaviour; the wheel adjusts the value.
- popup open  -> dropdown lists are separate widgets, so scrolling them still works.

WHY SUBCLASSES AND NOT AN APPLICATION-LEVEL EVENT FILTER
--------------------------------------------------------
``QComboBox.wheelEvent`` consumes the wheel event itself, before an
application-level ``installEventFilter`` on the QApplication ever sees it — the
filter runs too late to stop the value changing. Overriding ``wheelEvent`` on a
subclass is the only reliable interception point.

IMPORTANT: ``event.ignore()`` (not ``accept()``) is what lets the event bubble to
the parent scroll area. Swallowing it would freeze scrolling whenever the cursor
happened to be over a control.
"""
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QSlider,
    QSpinBox,
)

# Widget types whose value the wheel can silently change while scrolling past.
GUARDED_WIDGETS = (QComboBox, QAbstractSpinBox, QSlider)


class _WheelArmedControl:
    """Mixin: armour the wheel only after the user presses this control.

    MRO note: put this first in the bases so ``wheelEvent``/``mousePressEvent``
    resolve here, then ``super()`` continues into the Qt widget class.
    """

    def _init_wheel_arm(self):
        self._wheel_armed = False

    def mousePressEvent(self, event):
        # A press on the control is the deliberate signal to unlock the wheel.
        self._wheel_armed = True
        super().mousePressEvent(event)

    def wheelEvent(self, event):
        if not getattr(self, "_wheel_armed", False):
            # Don't handle it: Qt propagates the ignored event to the parent
            # scroll area, so the page scrolls instead of the value changing.
            event.ignore()
            return
        super().wheelEvent(event)


class GuardedComboBox(_WheelArmedControl, QComboBox):
    """QComboBox whose wheel only changes the value after it has been clicked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_wheel_arm()


class GuardedSpinBox(_WheelArmedControl, QSpinBox):
    """QSpinBox whose wheel only changes the value after it has been clicked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_wheel_arm()


class GuardedDoubleSpinBox(_WheelArmedControl, QDoubleSpinBox):
    """QDoubleSpinBox whose wheel only changes the value after it has been clicked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_wheel_arm()


class GuardedSlider(_WheelArmedControl, QSlider):
    """QSlider whose wheel only changes the value after it has been clicked."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_wheel_arm()


class _WheelGuard(QObject):
    """Application-level fallback for guarded widgets instantiated directly.

    Primary protection is the subclasses above; this catches anything that
    constructs a bare ``QComboBox``/``QAbstractSpinBox``/``QSlider`` so a
    forgotten call site degrades to "slightly annoying" rather than "silent
    data corruption".
    """

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel and isinstance(obj, GUARDED_WIDGETS):
            if not getattr(obj, "_wheel_armed", False):
                event.ignore()
        return False


def install_wheel_guard(app: QApplication):
    """Install the application-level fallback filter for the lifetime of ``app``."""
    guard = _WheelGuard(app)
    app.installEventFilter(guard)
    app._wheel_guard = guard  # retain so Qt's pointer stays valid
    return guard
