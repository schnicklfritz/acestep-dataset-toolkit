"""Static check for dangling ``self.<attr>`` references.

WHY THIS EXISTS
---------------
Commit 15520af removed the Studio's "Run AI Captioner" button
(``self.run_ai_btn = QPushButton(...)``) but left three references behind:

    start_ai_captioning()  -> self.run_ai_btn.setEnabled(False)
    on_caption_finished()  -> self.run_ai_btn.setEnabled(True)
    on_worker_error()      -> self.run_ai_btn.setEnabled(True)

Nothing failed, because the button's own wiring went too, so those lines were
unreachable. Two weeks later the Caption tab started calling
``start_ai_captioning()`` and every caption action raised AttributeError.

A missing attribute is not a syntax error and not an import error -- it is only
found by executing that exact path. This test finds it by analysis instead.

HOW IT WORKS
------------
1. Collect every attribute assignment in the app's own packages, from ANY of the
   forms used to attach widgets:
       self.x = ...        (inside a class)
       manager.x = ...     (ui/settings_tab.py builds onto the manager)
       parent.x = ...
       setattr(obj, "x", ...)
   Scanning the whole app matters: extracted UI modules assign widgets onto the
   manager, so a check that only looked at dataset_manager.py would report every
   extracted widget as dangling.
2. Collect every ``self.<attr>`` usage in dataset_manager.py.
3. Assert usage is a subset of assignment, minus an explicit allowlist.

Tests are EXCLUDED from the assignment scan: a test doing ``mock.thing = 1``
would otherwise make a genuinely dangling name look defined.
"""
import ast
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Directories that make up the application (not tests, not tooling).
APP_DIRS = ("", "ui", "modules", "workers", "core", "kernels", "scripts")

# Attributes set by means this scan cannot see (dynamic lookup, kwargs, Qt
# internals). Every entry needs a reason -- an allowlist without one is how a
# real dangling reference gets buried.
#
# Currently EMPTY: everything the class references is either assigned in the
# app's own source or inherited from Qt (see _qt_inherited_names).
ALLOWLIST = set()


def _qt_inherited_names():
    """Every public name provided by Qt's base classes.

    ``self.resize(...)``, ``self.setWindowTitle(...)`` and friends are inherited
    from QMainWindow, not assigned anywhere in this repo. Treating them as
    unknown would flag them as dangling.
    """
    try:
        from PySide6.QtWidgets import QMainWindow
    except ImportError:       # pragma: no cover - PySide6 is a hard dependency
        return set()
    return {n for n in dir(QMainWindow) if not n.startswith("__")}


def _python_files():
    for sub in APP_DIRS:
        d = os.path.join(ROOT, sub) if sub else ROOT
        if not os.path.isdir(d):
            continue
        for name in sorted(os.listdir(d)):
            if name.endswith(".py"):
                yield os.path.join(d, name)


def _collect_assigned_attributes():
    """Every attribute name the app can legitimately reference as ``self.<name>``.

    Three sources, because a class exposes several kinds of name through
    ``self``:

    * ``self.x = ...`` / ``manager.x = ...`` / ``setattr(obj, "x", ...)``
      -- widget and state attachment, including from extracted UI modules
    * ``def x(self): ...``  -- methods, which are also reached via ``self.x``
    * plain ``x = ...`` at module or class scope -- constants such as
      ``_MANUAL_COLS``

    Deliberately over-collects rather than risk false positives: a broad set here
    only means a name is considered known, whereas a narrow set would flag every
    method and every class constant as dangling.
    """
    assigned = set()
    for path in _python_files():
        try:
            with open(path, encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=path)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # methods
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned.add(node.name)

            targets = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Attribute):
                    assigned.add(target.attr)
                elif isinstance(target, ast.Name):
                    assigned.add(target.id)          # module/class constants
                elif isinstance(target, ast.Tuple):  # self.a, self.b = ...
                    for elt in target.elts:
                        if isinstance(elt, ast.Attribute):
                            assigned.add(elt.attr)
                        elif isinstance(elt, ast.Name):
                            assigned.add(elt.id)

            # setattr(obj, "name", value)
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "setattr"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)):
                assigned.add(node.args[1].value)
    return assigned


def _collect_self_usages(path):
    """Every ``self.<name>`` read or write inside ``path``."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    used = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "self"):
            used.add(node.attr)
    return used


@pytest.fixture(scope="module")
def assigned():
    return _collect_assigned_attributes()


@pytest.fixture(scope="module")
def self_usages():
    return _collect_self_usages(os.path.join(ROOT, "dataset_manager.py"))


class TestAssignmentScanIsReal:
    """Guard the guard: a scan that finds nothing would pass vacuously."""

    def test_scan_finds_many_assignments(self, assigned):
        assert len(assigned) > 100, "assignment scan is not seeing the app"

    def test_scan_finds_widgets_defined_in_extracted_ui_modules(self, assigned):
        # These live in ui/*.py, not dataset_manager.py. If they are missing the
        # scan is not covering the extracted modules, and every extracted widget
        # would be reported as dangling.
        for name in ("caption_blend_slider", "moss_track_picker",
                     "caption_backend_combo"):
            assert name in assigned, f"scan missed {name}"

    def test_scan_finds_many_usages(self, self_usages):
        assert len(self_usages) > 100

    def test_a_removed_attribute_is_genuinely_unassigned(self, assigned):
        # Documents the historical bug: run_ai_btn really is unassigned, so the
        # check below would have caught it rather than passing vacuously.
        assert "run_ai_btn" not in assigned


class TestNoDanglingAttributes:
    def test_every_self_usage_is_assigned_somewhere(self, assigned, self_usages):
        dangling = sorted(self_usages - assigned - _qt_inherited_names() - ALLOWLIST)
        assert not dangling, (
            "dataset_manager.py uses self.<attr> that nothing ever assigns:\n  "
            + "\n  ".join(dangling)
            + "\n\nThis is the run_ai_btn bug (widget removed, references left "
              "behind). Either assign it, guard the access with getattr, or add "
              "it to ALLOWLIST with a reason."
        )

    def test_allowlist_entries_are_still_used(self, self_usages):
        # An allowlist that grows stale is how a real bug hides. Every entry
        # must still correspond to something the class actually references.
        for name in ALLOWLIST:
            assert name in self_usages, (
                f"ALLOWLIST entry {name!r} is no longer used -- remove it"
            )

