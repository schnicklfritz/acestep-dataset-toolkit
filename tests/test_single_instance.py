"""One window, one writer: a second launch must not open a second editor.

WHY THIS EXISTS
---------------
Every save rewrites the whole dataset JSON from the in-memory copy. With two
windows open, the second window's stale copy overwrites the first window's work
on its next save — captions and lyrics written in one window vanish. The save
path cannot fix that: it can make a single write atomic, but it cannot merge two
whole copies of the dataset. The only place to close the race is at startup.

What is pinned here:

  * a second launch detects the running instance and is told to exit;
  * with nothing listening, the launch proceeds (becomes the primary);
  * ``main.py`` imports cleanly, so the guard is actually wired, not just
    defined and unused.
"""
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def socket_name(tmp_path, monkeypatch):
    """A unique, writable socket name so the test never talks to a real app."""
    import main

    name = str(tmp_path / "instance")
    monkeypatch.setattr(main, "_instance_socket_path", lambda: name)
    return name


def test_nothing_listening_means_no_running_instance(socket_name):
    import main

    assert main._activate_running_instance(socket_name) is False


def test_the_first_launch_becomes_the_primary(qapp, socket_name):
    import main

    primary = main._single_instance_guard(qapp)
    assert primary is not None, "the first launch must become the primary"


def test_a_second_launch_detects_the_running_instance(qapp, socket_name):
    import main

    primary = main._single_instance_guard(qapp)
    assert primary is not None
    # The second launch's client reaches the primary's socket, so it is told an
    # instance is up and the caller exits instead of opening a second window.
    assert main._activate_running_instance(socket_name) is True
    assert main._single_instance_guard(qapp) is None


def test_main_imports_and_exposes_the_guard():
    import main

    assert callable(main._single_instance_guard)
