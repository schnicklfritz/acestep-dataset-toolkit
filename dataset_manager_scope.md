# Dataset Toolkit: Practical Scope and Build Plan

## Guiding Rule

> Do not add work unless it materially improves dataset quality, training reliability, or the user's time-to-result.

The toolkit is a dataset-preparation assistant for ACE-Step LoRA / LoKR workflows. It should identify high-impact errors, automate reliable tasks, preserve user control, and avoid creating new model-training semantics unless the stock training pipeline consumes them.

## Build Now

### Dataset readiness scan

Provide one visible **Scan Audio + Fill Metadata** action that performs the following in a background worker:

- Verify every referenced local audio file exists and can be decoded.
- Read duration, codec/container, sample rate, bit depth where available, channel count, and file size.
- Flag clipped audio, excessive leading/trailing silence, near-silent files, very short files, corrupt/unreadable files, and probable accidental concatenations.
- Detect likely low-quality/lossy outliers based on codec, bitrate, and spectral indicators where available.
- Compare each file with the dataset distribution rather than imposing a one-size-fits-all rule.
- Show a per-track health status and an actionable dataset-level readiness report.

### Homogeneity audit

Flag high-impact inconsistencies:

- Mixed sample rates and channel counts.
- Large loudness spread.
- Extreme clipping.
- Missing/corrupt paths.
- Severe source-quality outliers.
- Very short or potentially incomplete files.

The audit should distinguish between:

- **Ready**: no material issue detected.
- **Normalize**: reversible format or loudness mismatch.
- **Review**: uncertain semantic/metadata result requiring a human listener.
- **Replace or remove**: missing, corrupt, badly clipped, or clearly unsuitable source.

### Optional DSP normalization

Offer—not force—normalization for reversible issues:

- Resample to the user-selected target format.
- Normalize loudness with an explicit target chosen for the training pipeline.
- Apply a safe peak ceiling.
- Preserve originals.
- Write derived files to a visible user-selected output directory, never only to a temporary directory.
- Re-scan outputs after processing.

Normalization must clearly state that it cannot restore fidelity lost to lossy compression, clipping, or poor original sources.

### Metadata autofill and locking

Autofill values only when they can be measured or have adequate confidence:

- Duration: authoritative container/audio read; locked.
- Sample rate, channels, codec/container: authoritative file read; locked.
- BPM, key, and time signature: detected estimate with confidence; locked by default.
- Language and instrumental state: only when a suitable detector is configured; otherwise leave as unknown/unset.
- Genre and caption: AI suggestions, never claimed as factual measurements.

Provide visible controls in the track inspector:

- **Unlock detected metadata for this track**
- **Unlock all detected metadata**
- **Re-scan selected track**
- **Restore detected value**

A user override must never be silently overwritten by another scan. Store provenance when practical:

```json
{
  "bpm": 124,
  "bpm_source": "detected",
  "bpm_confidence": 0.86,
  "bpm_locked": true,
  "keyscale": "E minor",
  "keyscale_source": "user_override"
}
```

### Safe UI behavior

Do not use freely scrollable editable widgets for high-impact metadata without a lock state.

The following should be locked after detection or explicit approval:

- Language
- Instrumental state
- Prompt override
- Genre
- BPM
- Key scale
- Time signature
- Duration

Keep captions and formatted lyrics directly editable because those require human review.

## Do Not Build Yet

### Automatic song splitting at BPM changes

Do not split a track merely because tempo changes occur. Tempo variation can be core musical structure and may be desirable for the LoRA to learn.

For variable-tempo tracks:

- Keep the whole song by default.
- Omit BPM or use the trainer's Auto/N/A option when supported.
- Optionally store an app-only `tempo_variation: true` note for user review.
- Offer segmentation only at clear, user-reviewed musical boundaries or for genuinely concatenated recordings.

### Custom tempo-map model conditioning

A custom `tempo_regions` field is useful for the app's audit trail, but stock ACE-Step training will ignore it unless the training data loader, conditioning format, and generation interface are all modified. This is not a LoRA-only feature and has low current return on effort.

### Automatic web downloading or source acquisition

Do not download music or integrate scraping/downloader functions. The toolkit may state that a flagged track appears inconsistent or lossy and recommend replacing it with a lawfully acquired lossless source chosen by the user.

## Recommended UX

```text
Dataset Readiness
[ Scan Audio + Fill Metadata ] [ Audit Homogeneity ] [ DSP Normalize ]
Status: Review — 4 tracks need attention

Detected metadata is locked.
[ Unlock selected track ] [ Unlock all detected metadata ]
```

Per track:

```text
Health: Review
• 22.05 kHz MP3; inconsistent with the 44.1 kHz lossless majority
• Estimated BPM: 126 (confidence 0.58; possible half/double-time)
• Key: E minor (confidence 0.71)

Recommendation: Replace with a clean source if available. Normalization can align
format and loudness but cannot restore removed high-frequency content.
```

## Next Brainstorming Areas

After the core scan/locking/normalization flow is stable, compare professional annotation and dataset-preparation tools for high-value additions such as:

- Bulk find/replace and caption templates.
- Duplicate/near-duplicate audio detection.
- Waveform preview, region markers, and A/B listening.
- Dataset split management (train/validation/test) and manifest validation.
- Caption completeness and consistency checks.
- Undo/redo and change history.
- Metadata provenance and audit export.
- Batch actions with dry-run preview.
- Portable project paths and missing-file relinking.
