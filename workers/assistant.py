"""AI Assistant: DeepSeek-powered live help for the app.

The assistant answers questions about using the app using an embedded help
document, and can reason about the user's current dataset (summary provided at
request time). It uses the same DeepSeek key as the captioner aggregator.
"""
from PySide6.QtCore import QThread, Signal
from openai import OpenAI

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
3. Spatial Pipeline — L/R spatial analysis variant.
4. MVSEP / Kaggle Separator — stem separation (cloud backends).
5. Advanced Tools — DeepSeek master-prompt orchestration.
6. Appearance & Customization — themes, fonts, UI zoom.
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


def summarize_dataset(dataset, health_reports=None):
    """Build a compact text summary of the current dataset for the assistant.

    Includes per-track metadata (genre, BPM, key, language, time signature,
    instrumental flag, prompt style) plus the audio health flags from the last
    Scan & Fill (clipping, lossy cutoff, sample rate, issues).
    """
    if not dataset:
        return None
    samples = dataset.get("samples", []) or []
    meta = dataset.get("metadata", {}) or {}
    health_reports = health_reports or {}
    lines = [
        f"Dataset: {meta.get('name') or '(unnamed)'} | "
        f"{len(samples)} track(s) | tag: {meta.get('custom_tag') or 'none'} | "
        f"tag_position: {meta.get('tag_position') or 'prepend'} | "
        f"mode: {meta.get('instrumental_mode') or 'mixed'}",
    ]
    for i, s in enumerate(samples[:12], start=1):
        name = s.get("filename", f"Track {i}")
        sid = s.get("id", "")
        rep = health_reports.get(sid, {}) or {}
        flags = []
        if rep.get("is_clipping"):
            flags.append("clipping")
        if rep.get("has_lossy_cutoff"):
            flags.append("lossy-cutoff")
        if rep.get("sample_rate") and rep.get("sample_rate") != 44100:
            flags.append(f"{rep.get('sample_rate')}Hz")
        issues = rep.get("issues") or []
        if issues:
            flags.append("issues: " + "; ".join(issues))
        flag_str = f" | flags: {', '.join(flags)}" if flags else ""
        line = (
            f"  {i}. {name}"
            f" | genre: {s.get('genre') or '?'}"
            f" | bpm: {s.get('bpm') or '?'}"
            f" | key: {s.get('keyscale') or '?'}"
            f" | lang: {s.get('language') or '?'}"
            f" | ts: {s.get('timesignature') or '?'}"
            f" | {'instrumental' if s.get('is_instrumental') else 'vocal'}"
            f" | style: {s.get('prompt_style') or 'use_global'}"
            f"{flag_str}"
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

                _name, info, client = get_client(self.config)
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



