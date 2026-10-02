
import os
os.environ["GOOGLE_GENERATIVE_AI_API_KEY"] = AIzaSyCI-E5X-GG5u0n0c9eJMbD5Evo5VNLx6Yo

"""AI Assistant: DeepSeek-powered live help for the app.
The assistant answers questions about using the app using an embedded help
document, and can reason about the user's current dataset (summary provided at
request time). It uses the same DeepSeek key as the captioner aggregator.
"""
from PySide6.QtCore import QThread, Signal

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ---------------------------------------------------------------------------
# Live help document — this is what makes the assistant a "live help file".
# ---------------------------------------------------------------------------
APP_HELP_TEXT = """\
ACE-Step Dataset Toolkit — an all-in-one desktop app (PySide6) for preparing,
auditing, normalizing and auto-captioning ACE-Step training datasets.

== TABS ==
1. Dataset Studio — add/import audio tracks, per-track table, live quality gauge.
2. Structural Pipeline — full song structure analysis + instrument extraction.
3. MVSEP / Kaggle Separator — stem separation (cloud backends).
4. Advanced Tools — DeepSeek master-prompt orchestration.
5. Appearance & Customization — themes, fonts, UI zoom.
7. AI Assistant (this tab) — ask anything about the app or your dataset.

== CORE WORKFLOW ==
1. Dataset Studio: Add Audio (files or folder) — tracks appear in the table.
2. Scan & Fill — health audit: sample rate, channels, clipping, lossy cutoff,
   BPM/key confidence; populates metadata; the quality gauge shows penalties.
3. DSP Normalize — EBU R128 to -14 LUFS / 44.1 kHz; originals backed up;
   Undo/Redo + A/B compare available.
4. AI Caption — backends: Kaggle Cloud GPU (Qwen2.5-Omni 11B captioner),
   Local ACE-Step (CUDA), Custom endpoint, DeepSeek, or Local rule engine.
5. Validate & Save JSON — outputs the ACE-Step manifest.

== STRUCTURAL PIPELINE ==
- Scope: All Tracks / Tracks Missing Captions / Selected Tracks (by number or
  list). Run separates stems (import or MVSEP), finds structural boundaries
  (Lyrics tags or MFCC agglomerative), captions sections, aggregates via
  DeepSeek into a master caption.
- Humanization Preset is free-form (type any artist/style).
- "Detect via Captioner" finds the instruments: it cuts the track at structural
  tags (short chunks name instruments precisely), captions each section with an
  instruments-only prompt, then asks DeepSeek which instrument-specific MVSEP
  models to run. Recommended models auto-flow into the pipeline.
- Instrument-specific extraction checkbox enables per-instrument stems.

== MVSEP / KAGGLE SEPARATOR ==
- Backend: MVSEP Cloud API (live algorithm list — always current models) or
  Kaggle GPU (Meta Demucs: htdemucs_ft / htdemucs / htdemucs_6s).
- Full separation runs a first stage (default BS PolarFormer 124-band, which
  re-synthesizes the instrumental and prevents artifacts/clipping) then the
  selected model on the instrumental. First stage is a dropdown — change it to
  any live algorithm.
- "Add stems to Dataset Studio" sends finished stems into the dataset.

== SETTINGS & SECRETS ==
- Credentials (MVSEP key, DeepSeek key, Kaggle key) are stored encrypted in the
  OS keyring (or secrets.enc fallback), never in settings.json. If a key is
  missing when needed, a popup appears: either save it securely or send it for
  the session only.
- Kaggle Username/Key, DeepSeek key, MVSEP key, custom endpoint URL are set in
  the Appearance & Customization tab (Cloud & Execution Endpoints).

== TROUBLESHOOTING ==
- "All tracks already have captions" → switch scope to All Tracks or Selected.
- Kaggle job fails → check Kaggle Username/Key in Settings; ensure internet.
- Quality warnings → use 'I Know What I'm Doing' bypass to export anyway.
- No instrument models returned → the caption may mention no instruments; try
  'Detect via Captioner' on a track with clear instrumentation.
"""


def build_system_prompt(help_text, dataset_summary):
    """System prompt: app docs + live dataset context."""
    return (
        "You are the built-in AI assistant for 'ACE-Step Dataset Toolkit', a "
        "PySide6 desktop app for preparing, auditing, normalizing and captioning "
        "ACE-Step training datasets. Use the app documentation below to answer "
        "questions about how to use the app. If the user asks about their dataset "
        "or the instruments in it, use the dataset summary. Be concise, practical, "
        "and specific. If asked to identify instruments, reason from the provided "
        "captions/sections and known studio practice.\n\n"
        "=== APP DOCUMENTATION ===\n"
        f"{help_text}\n\n"
        "=== CURRENT DATASET ===\n"
        f"{dataset_summary or '(no dataset loaded yet)'}"
    )


def summarize_dataset(dataset):
    """Build a compact text summary of the current dataset for the assistant.

    Includes per-track metadata (genre, BPM, key, language, time signature,
    instrumental flag, prompt style).
    """
    if not dataset:
        return None
    samples = dataset.get("samples", []) or []
    meta = dataset.get("metadata", {}) or {}
    lines = [
        f"Dataset: {meta.get('name') or '(unnamed)'} | "
        f"{len(samples)} track(s) | tag: {meta.get('custom_tag') or 'none'} | "
        f"tag_position: {meta.get('tag_position') or 'prepend'} | "
        f"mode: {meta.get('instrumental_mode') or 'mixed'}",
    ]
    for i, s in enumerate(samples[:12], start=1):
        name = s.get("filename", f"Track {i}")
        line = (
            f"  {i}. {name}"
            f" | genre: {s.get('genre') or '?'}"
            f" | bpm: {s.get('bpm') or '?'}"
            f" | key: {s.get('keyscale') or '?'}"
            f" | lang: {s.get('language') or '?'}"
            f" | ts: {s.get('timesignature') or '?'}"
            f" | {'instrumental' if s.get('is_instrumental') else 'vocal'}"
            f" | style: {s.get('prompt_style') or 'use_global'}"
        )
        cap = (s.get("caption") or "").strip().replace("\n", " ")[:120]
        if cap:
            line += f"\n       caption: {cap}"
        if s.get("detected_instruments"):
            line += f"\n       instruments: {s.get('detected_instruments')}"
        lines.append(line)
    if len(samples) > 12:
        lines.append(f"  … and {len(samples) - 12} more track(s).")
    return "\n".join(lines)


# Shared with the MCP server (kept headless — no Qt dependency here).
from modules.sound_profile import build_sound_profile  # noqa: E402


# ---------------------------------------------------------------------------
# Assistant tools (OpenAI function-calling) — the "plugin interface" that lets
# the assistant trigger app actions instead of only advising.
# ---------------------------------------------------------------------------
ASSISTANT_TOOLS = [
    {"type": "function", "function": {
        "name": "get_dataset_summary",
        "description": "Get the current dataset summary (tracks, metadata, health flags, captions).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "list_tracks",
        "description": "List all tracks in the dataset (index, filename, caption).",
        "parameters": {"type": "object", "properties": {"max_items": {"type": "integer", "description": "max tracks to list"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "lookup_instruments",
        "description": "Look up instruments for a track from the local database (filename match).",
        "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
    {"type": "function", "function": {
        "name": "audit_captions",
        "description": "Audit caption consistency (missing captions, instrument naming) across the dataset.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "validate_manifest",
        "description": "Validate the dataset manifest against the ACE-Step schema.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "scan_health",
        "description": "Run the health audit (Scan & Fill) on the dataset.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "detect_instruments",
        "description": "Start instrument detection for the selected track.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "get_dataset_sound_profile",
        "description": "Summarize the dataset's current sound (genres, BPM range, keys, instruments, vocal/instrumental mix, caption coverage).",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "curate_dataset",
        "description": "Recommend what to add so the dataset converges on a target sound. Pass a target_sound (artist/genre/mood); the tool returns the current sound profile + gap hints and you compose the curation plan. After proposing candidates, optionally call rockstar_lookup(song, artist) for any you want to flag for licensing verification.",
        "parameters": {"type": "object", "properties": {
            "target_sound": {"type": "string", "description": "e.g. 'Black Sabbath / doom blues, downtuned, slow'"}}, "required": ["target_sound"]}}},
    {"type": "function", "function": {
        "name": "rockstar_lookup",
        "description": "Check whether multitrack stems are known to exist for a song (community chart indices). Returns existence + references (titles/sites) only, never file links. Use to note which candidate songs have community multitracks available for licensing verification.",
        "parameters": {"type": "object", "properties": {
            "song": {"type": "string", "description": "song title"},
            "artist": {"type": "string", "description": "artist name (optional)"}},
            "required": ["song"]}}},
]

# ---------------------------------------------------------------------------
# Action tools — the ones that make the assistant *do* the task
# ---------------------------------------------------------------------------
# Read-only tools answer a question. These change the dataset (or stage work on
# disk), which is where the risk lives, so every description states exactly what
# it touches and what it will refuse:
#
#   * ``tracks`` is a list of filenames OR the numbers ``list_tracks`` printed.
#     A name is unambiguous, an index is a guess, so filenames are preferred.
#   * Every write goes through modules/assistant_actions.py, which holds the
#     field allowlist — a model cannot repoint ``audio_path``, rewrite an ``id``
#     or flip ``labeled`` to claim a human reviewed it.
#   * ffmpeg work is reported as a result line, never as a silent rewrite of the
#     training audio (originals are backed up first).
ACTION_TOOLS = [
    {"type": "function", "function": {
        "name": "set_track_metadata",
        "description": "Set ONE metadata field on one or more tracks. Writable fields: caption, genre, custom_tag, language, keyscale, timesignature, bpm, duration, is_instrumental, lyrics, formatted_lyrics. Reference tracks by filename (preferred) or by the number shown in list_tracks. Use for: 'mark every track instrumental', 'set the language to en', 'call this genre doom'.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "string"},
                       "description": "filenames or list_tracks numbers"},
            "field": {"type": "string", "description": "one of the writable fields above"},
            "value": {"type": "string", "description": "the new value; booleans accept 'true'/'false'"}},
            "required": ["tracks", "field", "value"]}}},
    {"type": "function", "function": {
        "name": "write_caption",
        "description": "Write a full ACE-Step caption onto tracks and report every schema violation it contains (front-loaded 5-12 keywords, a vocal descriptor, 2-3 flow sentences, no BPM/key/time-signature in the prose). Use when drafting or rewriting a caption, and fix and rewrite when violations come back.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "string"}},
            "caption": {"type": "string", "description": "the caption text"}},
            "required": ["tracks", "caption"]}}},
    {"type": "function", "function": {
        "name": "import_lyrics",
        "description": "Put a full lyric block onto tracks, writing formatted_lyrics, lyrics and raw_lyrics together so the export field cannot drift. Clears an instrumental flag rather than leaving the track contradictory. Use when you are given lyrics to attach to a track.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "string"}},
            "lyrics": {"type": "string", "description": "the lyric block, including [Section] markers"}},
            "required": ["tracks", "lyrics"]}}},
    {"type": "function", "function": {
        "name": "find_gaps",
        "description": "Audit the dataset for missing or omitted features: absent metadata, contradictions (instrumental with lyrics, lyrics without [Section] markers), caption schema violations, duplicate filenames, audio not on disk. Returns a numbered per-track report. Run this BEFORE planning work and again after it.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},

    {"type": "function", "function": {
        "name": "create_virtual_dataset",
        "description": "Draft concept-only tracks (no audio) so a dataset can be planned and its gaps found before anything is recorded or downloaded. Only allowlisted metadata is honoured; each track is flagged virtual and skipped by file-based steps and by export until real audio is attached.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "object", "properties": {
                "filename": {"type": "string"},
                "genre": {"type": "string"},
                "caption": {"type": "string"},
                "language": {"type": "string"},
                "bpm": {"type": "integer"},
                "keyscale": {"type": "string"},
                "is_instrumental": {"type": "boolean"}}}},
            "name": {"type": "string", "description": "optional dataset name"}},
            "required": ["tracks"]}}},
    {"type": "function", "function": {
        "name": "normalize_audio",
        "description": "Start EBU R128 loudness normalization (ffmpeg loudnorm) in the background. Normalized copies go to <target_dir>/normalized_audio and originals are copied to <target_dir>/originals_backup first, so the training audio is never overwritten. Returns the job it started.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "string"}},
            "target_dir": {"type": "string", "description": "output folder for normalized audio + backups"},
            "target_lufs": {"type": "number", "description": "target loudness, default -14.0"},
            "target_sr": {"type": "integer", "description": "sample rate, default 44100"}},
            "required": ["tracks", "target_dir"]}}},
    {"type": "function", "function": {
        "name": "stage_temp_mp3",
        "description": "Plan the ffmpeg command that converts tracks to a small scratch MP3 (stereo, 44.1 kHz, lossless source untouched) — used to preview a uniform upload before transcription or a Kaggle run. Returns the commands instead of running them, so they can be reviewed first.",
        "parameters": {"type": "object", "properties": {
            "tracks": {"type": "array", "items": {"type": "string"}},
            "target_dir": {"type": "string", "description": "where the scratch MP3s would go"},
            "bitrate": {"type": "string", "description": "e.g. 192k, default 192k"}},
            "required": ["tracks", "target_dir"]}}},

]

ASSISTANT_TOOLS = ASSISTANT_TOOLS + ACTION_TOOLS



class AssistantWorker(QThread):
    answer_ready = Signal(str)
    tool_requested = Signal(str, str, str)   # name, arguments-json, tool_call_id
    failed = Signal(str)

    def __init__(self, api_key, messages, tools=None, parent=None, config=None):
        super().__init__(parent)
        self.api_key = api_key
        self.config = config
        self.messages = messages
        self.tools = tools

    def run(self):
        try:
            if self.config is not None:
                from modules.llm_client import get_client

                _name, info, client = get_client(self.config, role="assistant")
                model = info.get("model") or "deepseek-chat"
            else:
                from openai import OpenAI

                client = OpenAI(api_key=self.api_key, base_url=DEEPSEEK_BASE_URL)
                model = "deepseek-chat"
            kwargs = dict(
                model=model,
                messages=self.messages,
                temperature=0.4,
                max_tokens=900,
            )
            if self.tools:
                kwargs["tools"] = self.tools

            # ROBUST SANITIZER: Handles both dicts and Pydantic/OpenAI objects
            if "messages" in kwargs and isinstance(kwargs["messages"], list):
                sanitized_list = []
                for msg in kwargs["messages"]:
                    # Convert OpenAI model objects to dict if necessary
                    if hasattr(msg, "model_dump"):
                        m_dict = msg.model_dump(exclude_unset=True)
                    elif hasattr(msg, "to_dict"):
                        m_dict = msg.to_dict()
                    elif isinstance(msg, dict):
                        m_dict = dict(msg)
                    else:
                        m_dict = {"role": getattr(msg, "role", "user"), "content": str(msg)}

                    # Strip all reasoning/thinking metadata fields that Groq rejects
                    m_dict.pop("reasoning_content", None)
                    m_dict.pop("reasoning", None)
                    m_dict.pop("thinking_blocks", None)

                    sanitized_list.append(m_dict)

                kwargs["messages"] = sanitized_list

            response = client.chat.completions.create(**kwargs)

            msg = response.choices[0].message
            if getattr(msg, "tool_calls", None):
                call = msg.tool_calls[0]
                self.tool_requested.emit(
                    call.function.name,
                    call.function.arguments or "{}",
                    call.id,
                )
                return
            self.answer_ready.emit((msg.content or "").strip())
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))



