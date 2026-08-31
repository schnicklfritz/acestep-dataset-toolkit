import os, json, math, struct, wave, subprocess
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
from PySide6.QtCore import QThread, Signal

class HealthAuditorWorker(QThread):
    progress = Signal(int, str)
    file_audited = Signal(str, dict)
    audit_completed = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, samples):
        super().__init__()
        self.samples = samples
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.audit_completed.emit({"quality_score": 100, "healthy": True, "reasons": []})
                return

            sample_rates = []
            channels_list = []
            lufs_list = []
            clipping_files = []
            lossy_files = []
            missing_files = []
            uncertain_bpm = []
            reports = {}

            for idx, s in enumerate(self.samples):
                if self._is_cancelled:
                    return

                sid = s.get("id", "")
                path = s.get("audio_path", "")
                fname = s.get("filename", "")

                if not path or not os.path.exists(path):
                    missing_files.append(fname)
                    rep = {
                        "status": "Missing",
                        "issues": ["Audio file missing from disk"],
                        "confidence": 1.0,
                        "penalty": 40
                    }
                    reports[sid] = rep
                    self.file_audited.emit(sid, rep)
                    continue

                rep = self._analyze_track(path, s)
                reports[sid] = rep
                self.file_audited.emit(sid, rep)

                if rep.get("sample_rate"):
                    sample_rates.append(rep["sample_rate"])
                if rep.get("channels"):
                    channels_list.append(rep["channels"])
                if rep.get("lufs") is not None:
                    lufs_list.append(rep["lufs"])
                if rep.get("is_clipping"):
                    clipping_files.append(fname)
                if rep.get("has_lossy_cutoff"):
                    lossy_files.append(fname)
                if rep.get("bpm_confidence", 1.0) < 0.65:
                    uncertain_bpm.append(fname)

                pct = int(100 * (idx + 1) / total)
                self.progress.emit(pct, f"Audited: {fname}")

            # Compute Global Quality Score & Penalties
            quality_score = 100
            reasons = []

            if missing_files:
                pen = min(40, len(missing_files) * 20)
                quality_score -= pen
                reasons.append(f"Missing Files (-{pen}%): {len(missing_files)} track(s) cannot be read on disk.")

            if clipping_files:
                pen = min(20, len(clipping_files) * 10)
                quality_score -= pen
                reasons.append(f"Digital Clipping (-{pen}%): {len(clipping_files)} track(s) exceed -0.1 dBFS ceiling.")

            if lossy_files:
                pen = min(20, len(lossy_files) * 8)
                quality_score -= pen
                reasons.append(f"Lossy Source Inconsistency (-{pen}%): {len(lossy_files)} track(s) have <192kbps / high-frequency cutoffs.")

            unique_sr = list(set(sample_rates))
            if len(unique_sr) > 1:
                quality_score -= 10
                reasons.append(f"Mixed Sample Rates (-10%): Dataset mixes {unique_sr} Hz.")

            unique_ch = list(set(channels_list))
            if len(unique_ch) > 1:
                quality_score -= 10
                reasons.append(f"Mismatched Channels (-10%): Dataset mixes {unique_ch} channels (Mono & Stereo).")

            lufs_spread = 0.0
            if lufs_list:
                lufs_spread = max(lufs_list) - min(lufs_list)
                if lufs_spread > 5.0:
                    quality_score -= 15
                    reasons.append(f"Loudness Spread (-15%): Volume variation across tracks is {lufs_spread:.1f} dB.")

            if total < 10:
                quality_score -= 15
                reasons.append(f"Small Dataset (-15%): Current size ({total} tracks) is under recommended 10+ samples.")

            quality_score = max(5, min(100, quality_score))

            summary = {
                "quality_score": quality_score,
                "healthy": quality_score >= 80,
                "reasons": reasons,
                "total_audited": total,
                "unique_sample_rates": unique_sr,
                "unique_channels": unique_ch,
                "lufs_spread": lufs_spread,
                "clipping_count": len(clipping_files),
                "lossy_count": len(lossy_files),
                "missing_count": len(missing_files)
            }

            self.audit_completed.emit(summary)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def _analyze_track(self, path, sample_data):
        sr = 44100
        channels = 2
        dur = 0.0
        is_clipping = False
        has_lossy_cutoff = False
        lufs_est = -14.0
        bpm_detected = sample_data.get("bpm", 0)
        bpm_confidence = float(sample_data.get("bpm_confidence", 0.85))
        key_detected = sample_data.get("keyscale", "")
        key_confidence = float(sample_data.get("key_confidence", 0.80))
        issues = []

        try:
            if path.lower().endswith(".wav"):
                with wave.open(path, "rb") as wf:
                    sr = wf.getframerate()
                    channels = wf.getnchannels()
                    nframes = wf.getnframes()
                    dur = nframes / float(sr) if sr > 0 else 0
                    sampwidth = wf.getsampwidth()

                    frames_to_read = min(nframes, sr * 15)
                    raw_data = wf.readframes(frames_to_read)
                    if sampwidth == 2 and raw_data:
                        fmt = f"<{len(raw_data)//2}h"
                        samples = struct.unpack(fmt, raw_data)
                        max_val = max(abs(s) for s in samples) if samples else 0
                        rms = math.sqrt(sum(s*s for s in samples) / len(samples)) if samples else 0

                        if max_val >= 32700:
                            is_clipping = True
                            issues.append("Digital clipping (> -0.1 dBFS)")
                        if rms > 0:
                            lufs_est = 20 * math.log10(rms / 32768.0) - 3.0
            else:
                cmd = ["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate,channels,duration,bit_rate", "-of", "json", path]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0:
                    meta = json.loads(res.stdout)
                    streams = meta.get("streams", [{}])[0]
                    sr = int(streams.get("sample_rate", 44100))
                    channels = int(streams.get("channels", 2))
                    dur = float(streams.get("duration", 0))
                    bitrate = int(streams.get("bit_rate", 0))
                    if bitrate > 0 and bitrate < 192000:
                        has_lossy_cutoff = True
                        issues.append(f"Lossy stream compression ({bitrate//1000} kbps)")

        except Exception:
            pass

        if sr not in (44100, 48000):
            issues.append(f"Non-standard sample rate ({sr} Hz)")
        if channels == 1:
            issues.append("Mono recording (Stereo recommended)")
        if dur > 0 and dur < 10:
            issues.append("Short track (< 10s)")

        if not bpm_detected or not key_detected:
            # Compute real BPM/key instead of falling back to placeholders.
            try:
                from modules.tagger import analyze_audio
                tags = analyze_audio(path)
                if not bpm_detected and tags.get("bpm"):
                    bpm_detected = tags["bpm"]
                    bpm_confidence = 0.9
                if not key_detected and tags.get("key"):
                    key_detected = tags["key"]
                    key_confidence = 0.9
            except Exception:  # noqa: BLE001 — tagging must never break the audit
                pass
        if not bpm_detected:
            bpm_detected = 0
            bpm_confidence = 0.4
        if not key_detected:
            key_detected = ""
            key_confidence = 0.4

        status = "Healthy" if not issues else "Warning"
        return {
            "status": status,
            "sample_rate": sr,
            "channels": channels,
            "duration": round(dur, 2),
            "is_clipping": is_clipping,
            "has_lossy_cutoff": has_lossy_cutoff,
            "lufs": lufs_est,
            "bpm_detected": bpm_detected,
            "bpm_confidence": bpm_confidence,
            "key_detected": key_detected,
            "key_confidence": key_confidence,
            "issues": issues
        }

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL DspNormalizerWorker
# ============================================================================
