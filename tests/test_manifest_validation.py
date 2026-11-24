"""Manifest validation: the checks run over a dataset before it is trained on.

WHY THE UNRESOLVABLE-ROW CHECK EXISTS
-------------------------------------
Two different documents are called "the dataset":

  * the app's PROJECT FILE (``dataset.json`` written by ``save_dataset``) — a
    superset that keeps internal bookkeeping and ``virtual: True`` placeholder
    rows, because the app reads them back;
  * the TRAINING MANIFEST (``export_json``) — the official shape, with the
    internal keys and the placeholders removed (``for_export``).

Inside the app a placeholder is legitimate: the lyric-gap audit and the captioner
both use those rows. An export strips the ``virtual`` flag along with the other
internal fields, so a placeholder that ever reached a manifest is invisible as a
placeholder and shows up only as a row with no ``filename`` and no ``audio_path``
— which the trainer cannot resolve. That is the one case worth flagging, and it
is exactly what these tests pin in both directions: quiet on a project file,
loud on a manifest that leaked a row.
"""
from modules.dataset_schema import new_dataset, new_sample
from modules.manifest_validation import validate_manifest

# Wording is not machine-scored any more; this is the shape upstream's own LoRA
# tutorial uses for a caption.
PROSE_CAPTION = "A high-energy J-pop track with synthesizer leads and fast tempo."


def _valid_dataset():
    ds = new_dataset(name="valid")
    ds["samples"] = [new_sample(
        id="aaa111", filename="a.mp3", audio_path="./a.mp3", genre="Rock",
        caption=PROSE_CAPTION,
    )]
    return ds


def _issues(ds):
    return [i for i in validate_manifest(ds)]


def test_a_prose_caption_is_not_flagged():
    """Caption wording is judged by a human/LLM, never by this validator.

    Regression: a dense-tag schema imported from the song-generation prompt used
    to reject a perfectly good descriptive caption at export time.
    """
    assert _issues(_valid_dataset()) == []


def test_a_complete_dataset_has_no_issues():
    assert _issues(_valid_dataset()) == []


def test_a_missing_filename_and_audio_path_is_flagged():
    """The leaked-placeholder shape: an anonymous row."""
    ds = _valid_dataset()
    ds["samples"][0]["filename"] = ""
    ds["samples"][0]["audio_path"] = ""
    issues = _issues(ds)
    assert any("no filename or audio_path" in i for i in issues)


def test_a_project_file_placeholder_is_not_flagged():
    """In-app, a ``virtual`` placeholder is legitimate — it must stay quiet.

    This is the half that makes the check usable: if validating the app's own
    dataset complained about every placeholder, the message would be noise and
    the real leak would be lost in it.
    """
    ds = new_dataset(name="project_file")
    ds["samples"] = [
        new_sample(id="aaa111", filename="a.mp3", audio_path="./a.mp3", genre="Rock",
                   caption=PROSE_CAPTION),
        new_sample(id="bbb222", filename="virtual_01_doom.wav", genre="Doom",
                   caption=PROSE_CAPTION, virtual=True),
    ]
    assert not [i for i in _issues(ds) if "no filename or audio_path" in i]


def test_a_real_row_with_a_pathed_filename_is_not_flagged():
    """``filename`` is not required to be bare — only to resolve."""
    ds = _valid_dataset()
    ds["samples"][0]["filename"] = "songs/a.mp3"
    ds["samples"][0]["audio_path"] = ""
    assert not [i for i in _issues(ds) if "no filename or audio_path" in i]
