"""Offline validation of kernels/caption_kernel.py's diagnostics.

A Kaggle run costs 10+ minutes and a multi-GB download, and the kernel used to be
SILENT for its first ~5 minutes (pip install, then the model download) — so
"nothing is happening" and "it is working" were indistinguishable from outside.
That is a real failure mode: the user reported exactly that ("it won't start, no
errors").

These tests pin the two fixes:

1. A preflight that prints python, GPU, INTERNET reachability, disk and the
   resolved audio folder BEFORE anything slow happens.
2. pip failures are REPORTED. ``check=False`` used to swallow them, which is how
   an Internet-off session failed with no error at all.
"""
import ast
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_caption_notebook import fill_kernel                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = os.path.join(ROOT, "kernels", "caption_kernel.py")


def _source():
    with open(KERNEL, encoding="utf-8") as fh:
        return fh.read()


def test_preflight_runs_before_the_install():
    """The whole point is that it prints before anything slow or silent."""
    src = _source()
    assert src.index("_preflight()") < src.index("_install()")


def test_preflight_reports_gpu_internet_and_disk():
    src = _source()
    assert "[caption] gpu" in src
    assert "[caption] internet" in src
    assert "socket.create_connection" in src
    assert "[caption] free disk" in src


def test_preflight_warns_loudly_about_a_missing_gpu_or_internet():
    src = _source()
    assert "!! NO GPU" in src
    assert "NOT REACHABLE" in src
    assert "Turn Internet ON" in src


def test_pip_failures_are_reported_not_swallowed():
    """check=False alone made an Internet-off session fail with no error."""
    src = _source()
    assert "proc.returncode != 0" in src
    assert "pip FAILED" in src


def test_audio_folder_and_file_count_are_printed():
    src = _source()
    assert "[caption] audio folder" in src
    assert "[caption] audio files" in src
    assert "NO AUDIO FOUND" in src
    assert "only ONE file found" in src


def test_model_source_and_load_time_are_printed():
    src = _source()
    assert "[caption] model source" in src
    assert "model loaded" in src
    assert "time.time() - _load_started" in src


def test_every_diagnostic_is_flushed():
    """Kaggle buffers stdout; an unflushed print looks like a hang."""
    tree = ast.parse(fill_kernel(_source()))
    diagnostics = [
        ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "print"
        and "[caption]" in ast.unparse(node)
    ]
    assert len(diagnostics) >= 12
    missing = [call for call in diagnostics if "flush=True" not in call]
    assert not missing, missing


def test_the_schema_reaches_the_kernel():
    """The system turn must carry the ACE-Step 1.5XL schema."""
    src = _source()
    assert "SYSTEM_PROMPT" in src
    assert '{"role": "system"' in src


def test_the_log_carries_the_full_caption_not_a_preview():
    """A 100-char preview made good captions look cut off in the Kaggle log."""
    src = _source()
    assert "caption[:100]" not in src
    assert 'print("OK", f.name, caption, flush=True)' in src


def test_the_kernel_documents_the_staging_folder_and_versioned_dataset():
    """The dataset is the app-uploaded staging folder, updated as a version."""
    src = _source()
    assert "STAGING folder" in src
    assert "VERSION" in src


def test_the_audio_limit_is_documented_as_the_only_truncation():
    src = _source()
    assert "MAX_AUDIO_DURATION" in src
    assert "0 = whole file" in src
    assert "[caption] limits" in src


class TestAudioDiscovery:
    """Kaggle mounts private datasets in more than one place.

    A real run found nothing because the dataset landed at
    /kaggle/input/datasets/<owner>/<slug>/ while the kernel looked in
    /kaggle/input/<slug>/. kernels/stem_separation_kernel.py solved this first and
    kernels/moss_caption_kernel.py copied it; THIS kernel kept the single path and
    then wrote a valid-looking {"results": []}, which reached the app as "no
    captions came back" and hid the cause for days.
    """

    def test_walks_the_tree_instead_of_one_folder(self):
        src = _source()
        assert "def _walk_for_audio(base):" in src
        assert "os.walk(base)" in src

    def test_falls_back_to_the_whole_input_mount(self):
        src = _source()
        assert '_walk_for_audio("/kaggle/input")' in src
        assert "was empty; found audio elsewhere" in src

    def test_does_not_assume_the_old_mount_layout(self):
        # The old code did Path(AUDIO_FOLDER).rglob("*") only.
        assert "Path(AUDIO_FOLDER)" not in _source()

    def test_no_audio_is_a_hard_failure_not_an_empty_result(self):
        """A zero-track result is indistinguishable from "the model returned
        nothing", which is exactly what made this look like a model problem."""
        src = _source()
        assert "NO AUDIO FOUND" in src
        assert "No supported audio found under" in src
        assert "raise SystemExit(" in src


# ---------------------------------------------------------------------------
# runtime contracts: these functions are EXECUTED, not grepped
# ---------------------------------------------------------------------------

def _kernel_function(name, extra_globals=None):
    """Compile ONE function out of the kernel template and return it callable.

    WHY: every other test in this file greps the SOURCE TEXT, and a source-text
    assertion cannot see a TYPE mismatch. Porting ``_walk_for_audio`` from the MOSS
    kernel (whose caller uses ``os.path.basename``) into this one (whose callers use
    the Path API) satisfied every grep here and then crashed a real Kaggle run four
    minutes in, right after the model had loaded. Anything with a runtime contract
    is EXECUTED in this section instead.
    """
    tree = ast.parse(_source())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(extra_globals or {})
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<kernel>", "exec"),
         namespace)
    return namespace[name]


def test_walk_for_audio_returns_paths_the_kernel_can_use(tmp_path):
    """The callers use Path (``batch[0].name``, ``f.name``) — strings crashed."""
    from pathlib import Path

    walk = _kernel_function(
        "_walk_for_audio", {"os": os, "Path": Path, "SUPPORTED_FORMATS": {".mp3"}}
    )
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "song.mp3").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")

    found = walk(str(tmp_path))
    assert len(found) == 1
    # THE call that killed run ace-caption-2f8e84: a str has no .name.
    assert found[0].name == "song.mp3"


def test_window_math_covers_the_whole_song():
    """A 4-minute song must be TWO passes, not one 120 s slice that discards half."""
    windows = _kernel_function(
        "_windows_for", {"os": os, "_duration_seconds": lambda _path: 240.0}
    )
    spans = windows("x.mp3", 120)
    assert spans == [(0.0, 120.0), (120.0, 120.0)]
    # Coverage, not truncation: the spans tile the file end to end.
    assert sum(length for _offset, length in spans) == 240.0


def test_the_last_window_is_clamped_to_the_end():
    windows = _kernel_function(
        "_windows_for", {"os": os, "_duration_seconds": lambda _path: 300.0}
    )
    assert windows("x.mp3", 120) == [(0.0, 120.0), (120.0, 120.0), (240.0, 60.0)]


def test_a_sub_second_tail_does_not_become_its_own_pass():
    windows = _kernel_function(
        "_windows_for", {"os": os, "_duration_seconds": lambda _path: 240.3}
    )
    assert windows("x.mp3", 120) == [(0.0, 120.0), (120.0, 120.0)]


def test_a_short_track_is_a_single_pass_and_zero_disables_windowing():
    windows = _kernel_function(
        "_windows_for", {"os": os, "_duration_seconds": lambda _path: 90.0}
    )
    assert windows("x.mp3", 120) == [(0, 0)]      # already fits one pass
    assert windows("x.mp3", 0) == [(0, 0)]        # 0 = whole file, single pass


def test_windows_are_never_batched_into_one_forward_pass():
    """The 32.61 GiB lesson, as an EXECUTED rule.

    Two 120 s clips in one audio forward pass asked a 14.56 GiB T4 for 32.61 GiB
    (the audio tower's attention is quadratic in the audio tokens), which killed
    run ace-caption-b34710 after the model had loaded. One clip per pass is the
    profile that has always worked here.
    """
    batches = _kernel_function("_pass_batches", {})
    assert batches(["a", "b", "c"]) == [["a"], ["b"], ["c"]]
    assert batches([]) == []
    assert batches(["only"]) == [["only"]]


class TestWholeSongWiring:
    """Source assertions for the parts that only exist on Kaggle."""

    def test_the_kernel_covers_whole_songs_by_default(self):
        src = _source()
        assert "WHOLE_SONG = {{WHOLE_SONG}}" in src
        assert "if WHOLE_SONG:" in src
        assert "def _windows_for(path, pass_sec):" in src
        assert "def _merge_part_captions(name, spans, parts):" in src

    def test_the_parts_are_merged_into_one_caption(self):
        src = _source()
        assert "_merge_part_captions(path.name, spans, parts)" in src
        assert "Merge them into a SINGLE caption" in src

    def test_a_failed_or_empty_merge_still_yields_a_caption(self):
        # Never lose a track: the first pass alone is a valid opening caption.
        src = _source()
        assert "if not caption.strip():" in src
        assert "caption = parts[0]" in src

    def test_the_single_pass_path_is_still_available(self):
        # OFF must keep working exactly as before (one pass, batch as configured).
        src = _source()
        assert "for i in range(0, len(audio_files), BATCH_SIZE):" in src

    def test_the_windowed_path_goes_through_the_single_clip_batches(self):
        # Batching the windows is what asked for 32.61 GiB and OOM'd the run.
        src = _source()
        assert "def _pass_batches(clips):" in src
        assert "_pass_batches(clips)" in src


# ---------------------------------------------------------------------------
# static check on the FILLED kernel: a substituted literal is not a variable
# ---------------------------------------------------------------------------

def _unbound_names(source):
    """Names READ but never bound anywhere in the file (a crude F821).

    WHY: a NameError happens only at RUNTIME — on Kaggle, minutes in, after a
    multi-GB model load. ``MAX_AUDIO_DURATION`` existed only as a SUBSTITUTED
    LITERAL (``{{MAX_AUDIO_DURATION}}`` becomes ``120``), so referring to it as a
    variable killed the first whole-song run at its first track::

        spans = _windows_for(path, MAX_AUDIO_DURATION)
        NameError: name 'MAX_AUDIO_DURATION' is not defined

    Deliberately CONSERVATIVE: a name bound ANYWHERE — a nested function, a loop
    target, an import, a function argument — counts as bound, so this cannot fire on
    valid code. It only reports names that are read and never bound at all.
    """
    import builtins

    tree = ast.parse(source)
    bound, loaded = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                bound.add((alias.asname or alias.name).split(".")[0])
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            else:
                loaded.add(node.id)
    return sorted(loaded - bound - set(dir(builtins)))


def test_no_kernel_name_is_read_without_ever_being_bound():
    """Checked on the FILLED kernel: every {{X}} becomes a literal, not a name."""
    from scripts.build_caption_notebook import fill_kernel

    script = fill_kernel(_source())
    assert "{{" not in script, "fill_kernel left a placeholder behind"
    assert _unbound_names(script) == []


def test_the_unbound_name_check_would_have_caught_the_nameerror():
    """Guard the guard: a test that cannot fire is worse than no test."""
    broken = (
        "MAX_NEW_TOKENS = 512\n"
        "def _windows_for(path, pass_sec):\n"
        "    return [(0, pass_sec)]\n"
        "def run(path):\n"
        "    return _windows_for(path, MAX_AUDIO_DURATION)\n"
    )
    # MAX_AUDIO_DURATION is READ and bound NOWHERE — the exact Kaggle failure.
    assert _unbound_names(broken) == ["MAX_AUDIO_DURATION"]
    # ...and silent once it is bound, which is what the fixed kernel does.
    assert _unbound_names("MAX_AUDIO_DURATION = 120\n" + broken) == []

