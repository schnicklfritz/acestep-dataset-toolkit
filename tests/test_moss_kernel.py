"""Offline validation of kernels/moss_caption_kernel.py.

A Kaggle run costs 20+ minutes and a 17 GiB download, so the cheap mistakes are
caught here first. These tests never touch the network and never need a GPU.

They exist because the MOSS-Audio API has three silent-failure modes:

1. ``MossAudioProcessor.from_pretrained`` defaults ``enable_time_marker=False``
   while ``__init__`` defaults it to True. Omitting it loses timestamps with no
   error -- and timestamps are the whole reason we use MOSS for lyrics.
2. Every processor kwarg is read via ``kwargs.pop(..., default)``, so a typo is
   silently ignored rather than raising.
3. ``config.json`` declares bfloat16. Kaggle T4s have no native bfloat16, so an
   ignored dtype argument means a failed run rather than a slow one.

Plus one structural trap: the Hugging Face repo does NOT contain
``modeling_moss_audio.py`` and its ``auto_map`` has no ``AutoModel`` entry, so
loading by repo id cannot resolve the model class. The GitHub clone is not
optional.
"""
import ast
import os
import re

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_PATH = os.path.join(ROOT, "kernels", "moss_caption_kernel.py")

# The audio encoder's positional embedding is fixed at max_source_positions=1500
# and audio_tokens_per_second=12.5, so one pass covers ~120 s. Hard limit, not
# a tunable.
MAX_AUDIO_SECONDS = 120

REQUIRED_PLACEHOLDERS = {
    "{{AUDIO_DATASET_PATH}}",
    "{{MODEL_ID}}",
    "{{STYLE_PROMPT}}",
    "{{LYRICS_PROMPT}}",
    "{{MAX_NEW_TOKENS}}",
    "{{CHUNK_SECONDS}}",
    "{{CUSTOM_TAG}}",
}

# Values the app would substitute at push time.
#
# CONVENTION (matches kernels/caption_kernel.py):
#   * A path placeholder is written INSIDE quotes in the kernel
#     (`AUDIO_FOLDER = "{{AUDIO_DATASET_PATH}}"`), so it is replaced with a BARE
#     value -- adding quotes here would produce `""/kaggle/input/audio""`.
#   * Prompt/tag placeholders stand alone and ARE replaced with a JSON string
#     literal, quotes included.
SUBSTITUTIONS = {
    "{{AUDIO_DATASET_PATH}}": "/kaggle/input/audio",
    "{{MODEL_ID}}": '"OpenMOSS-Team/MOSS-Audio-8B-Instruct"',
    "{{STYLE_PROMPT}}": '"Describe this music."',
    "{{LYRICS_PROMPT}}": '"Transcribe these lyrics."',
    "{{MAX_NEW_TOKENS}}": "1024",
    "{{CHUNK_SECONDS}}": "110",
    "{{CUSTOM_TAG}}": '""',
}


@pytest.fixture(scope="module")
def kernel_source():
    with open(KERNEL_PATH, encoding="utf-8") as f:
        return f.read()


def _top_level_function(source, name):
    """Return a whole top-level ``def name(...)`` block.

    Reviews the source as text, so this deliberately stops at the next
    top-level ``def`` rather than trying to parse (the raw kernel contains
    placeholders and will not parse).

    The naive ``def name\\(.*?return`` pattern is NOT enough: it stops at the
    first early ``return ""`` and silently truncates the body, which makes a
    presence assertion pass for the wrong reason.
    """
    match = re.search(rf"^def {re.escape(name)}\(.*?(?=^def |\Z)",
                      source, re.S | re.M)
    return match.group(0) if match else ""


@pytest.fixture(scope="module")
def substituted(kernel_source):
    """The kernel as the app would push it, ready for ast.parse."""
    text = kernel_source
    for placeholder, value in SUBSTITUTIONS.items():
        text = text.replace(placeholder, value)
    return text


class TestPlaceholders:
    def test_every_required_placeholder_is_present(self, kernel_source):
        found = set(re.findall(r"\{\{[A-Z_]+\}\}", kernel_source))
        missing = REQUIRED_PLACEHOLDERS - found
        assert not missing, f"kernel is missing placeholders: {sorted(missing)}"

    def test_no_unknown_placeholders(self, kernel_source):
        found = set(re.findall(r"\{\{[A-Z_]+\}\}", kernel_source))
        extra = found - REQUIRED_PLACEHOLDERS
        assert not extra, f"undocumented placeholders: {sorted(extra)}"

    def test_substitution_leaves_no_braces(self, substituted):
        assert "{{" not in substituted
        assert "}}" not in substituted


class TestKernelParses:
    def test_substituted_kernel_is_valid_python(self, substituted):
        ast.parse(substituted)

    def test_unsubstituted_placeholders_are_detectable(self, kernel_source):
        # NOTE: `MODEL_ID = {{MODEL_ID}}` is VALID Python syntax -- it parses as
        # a set containing a set. So a missed substitution does NOT raise
        # SyntaxError; it raises NameError at runtime, on Kaggle, 20 minutes in.
        # That is why the app must assert the braces are gone rather than rely
        # on the kernel failing loudly.
        assert not ast.parse.__doc__ is None  # keep the import meaningful
        assert "{{" in kernel_source, "expected placeholder braces in the raw kernel"
        assert re.search(r"\{\{[A-Z_]+\}\}", kernel_source)


class TestRepoCloneIsMandatory:
    """The HF repo cannot load the model; the GitHub clone is required."""

    def test_clones_the_official_repo(self, kernel_source):
        assert "github.com/OpenMOSS/MOSS-Audio" in kernel_source

    def test_imports_the_model_and_processor_classes(self, kernel_source):
        assert "from src." in kernel_source
        assert "MossAudioModel" in kernel_source
        assert "MossAudioProcessor" in kernel_source

    def test_repo_root_is_added_to_sys_path_not_src_itself(self, kernel_source):
        # src/ is a package using absolute `from src.x import y` internally, so
        # the REPO ROOT must be importable.
        assert "sys.path.insert(0, REPO_DIR)" in kernel_source
        assert 'REPO_DIR + "/src"' not in kernel_source


class TestSilentDefaultTraps:
    """Each of these fails silently at runtime if the kernel gets it wrong."""

    def test_time_marker_is_requested_explicitly(self, kernel_source):
        # from_pretrained defaults this to False; __init__ defaults to True.
        assert "enable_time_marker=True" in kernel_source

    def test_time_marker_is_asserted_not_trusted(self, kernel_source):
        # kwargs.pop() swallows typos, so verify the resolved value.
        assert re.search(
            r"enable_time_marker.*?\n.*?return|assert.*enable_time_marker",
            kernel_source, re.S,
        )

    def test_audio_data_is_cast_to_the_model_dtype(self, kernel_source):
        # MelConfig.mel_dtype is bfloat16; feeding that to an fp16 model on a
        # T4 is the failure this guards against.
        assert re.search(
            r'inputs\["audio_data"\]\s*=\s*inputs\["audio_data"\]\.to\(',
            kernel_source,
        )

    def test_audio_input_mask_is_built(self, kernel_source):
        assert 'inputs["audio_input_mask"]' in kernel_source
        assert "audio_token_id" in kernel_source

    def test_handles_both_dtype_kwarg_spellings(self, kernel_source):
        # transformers >= 4.56 accepts dtype=; older only torch_dtype=.
        assert "dtype=torch.float16" in kernel_source
        assert "torch_dtype=torch.float16" in kernel_source
        assert "TypeError" in kernel_source

    def test_transformers_is_pinned(self, kernel_source):
        # auto_docstring + models.qwen3 need a recent version and dtype= needs
        # >= 4.56. Kaggle's default is usually older.
        assert re.search(r"transformers==\d", kernel_source)

    def test_torch_is_not_reinstalled(self, kernel_source):
        # Clobbering Kaggle's CUDA build breaks the GPU stack.
        for call in re.findall(r"_pip\(([^)]*)\)", kernel_source):
            assert "torch==" not in call.replace("torchaudio", "")


class TestAudioLimit:
    """One pass covers ~120 s; longer tracks must be chunked."""

    def test_chunk_size_respects_the_hard_limit(self):
        assert int(SUBSTITUTIONS["{{CHUNK_SECONDS}}"]) < MAX_AUDIO_SECONDS

    def test_kernel_documents_the_limit(self, kernel_source):
        assert "1500" in kernel_source
        assert "120" in kernel_source

    def test_chunking_helper_splits_long_audio(self, substituted):
        # Exercise the real helper rather than trusting the source text.
        import numpy as np

        ns = {"MEL_SR": 16000}   # the helper reads this module-level constant
        fn = re.search(r"def _chunk\(audio, seconds\):.*?\n\n", substituted, re.S)
        assert fn, "_chunk helper not found"
        exec(fn.group(0), ns)

        audio = np.zeros(300 * 16000, dtype="float32")
        windows = ns["_chunk"](audio, 110)
        assert len(windows) == 3
        assert all(len(w[1]) <= 110 * 16000 for w in windows)

    def test_short_audio_is_not_chunked(self, substituted):
        import numpy as np

        ns = {"MEL_SR": 16000}
        fn = re.search(r"def _chunk\(audio, seconds\):.*?\n\n", substituted, re.S)
        exec(fn.group(0), ns)

        audio = np.zeros(30 * 16000, dtype="float32")
        windows = ns["_chunk"](audio, 110)
        assert len(windows) == 1
        assert windows[0][0] == 0


class TestDeterminism:
    def test_sampling_is_disabled(self, kernel_source):
        # Annotation should be reproducible; sampling makes re-runs differ.
        assert "do_sample=False" in kernel_source


class TestOutputContract:
    """The importer depends on this exact shape."""

    def test_writes_the_expected_json_path(self, kernel_source):
        assert "/kaggle/working/moss_out.json" in kernel_source

    def test_result_keys_match_the_importer(self, kernel_source):
        assert '"results"' in kernel_source
        for key in ('"file"', '"style"', '"lyrics"'):
            assert key in kernel_source

    def test_custom_tag_goes_on_the_caption_not_the_lyrics(self, kernel_source):
        # The trigger tag belongs on the style caption, not inside transcribed
        # lyrics (where it would be sung back).
        style_fn = _top_level_function(kernel_source, "style_for_track")
        lyrics_fn = _top_level_function(kernel_source, "lyrics_for_track")
        assert style_fn, "style_for_track not found"
        assert lyrics_fn, "lyrics_for_track not found"
        assert "CUSTOM_TAG" in style_fn
        assert "CUSTOM_TAG" not in lyrics_fn

