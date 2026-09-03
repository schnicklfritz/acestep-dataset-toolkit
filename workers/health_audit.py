# workers/health_audit.py
import os
import json
import subprocess
import re
from PySide6.QtCore import QThread, Signal

class HealthAuditorWorker(QThread):
    progress = Signal(int, str)
    file_audited = Signal(str, dict)   # Emitted during scan
    audit_completed = Signal(dict)     # Final summary with recommendation
    error_occurred = Signal(str)

    def __init__(self, samples, config=None):
        super().__init__()
        self.samples = samples
        self.config = config or {}
        # Tolerance: how far (in LU) a file can stray from the dataset average before warning
        self.lufs_deviation_tolerance = self.config.get("lufs_tolerance", 3.0)
        self.is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.audit_completed.emit({"healthy": True, "summary": "No files to check."})
                return

            # --- PASS 1: Collect raw data from every file ---
            reports = {}
            raw_lufs_values = []  # We'll use this to calculate the dataset average

            for idx, s in enumerate(self.samples):
                if self.is_cancelled:
                    return

                sid = s.get("id")
                path = s.get("audio_path")
                fname = s.get("filename", "")

                if not path or not os.path.exists(path):
                    sr, ch, dur, lufs = 0, 0, 0.0, None
                    issues = ["File missing"]
                    status = "Missing"
                else:
                    sr, ch, dur, lufs, issues = self._get_basic_audio_info(path)
                    status = "Warning" if issues else "Healthy"

                # Store report (without dataset-relative LUFS warnings yet)
                report = {
                    "id": sid,
                    "filename": fname,
                    "status": status,
                    "sample_rate": sr,
                    "channels": ch,
                    "duration": round(dur, 2),
                    "integrated_lufs": round(lufs, 2) if lufs is not None else None,
                    "issues": issues,
                }
                reports[sid] = report
                self.file_audited.emit(sid, report)

                if lufs is not None:
                    raw_lufs_values.append(lufs)

                pct = int(100 * (idx + 1) / total)
                self.progress.emit(pct, f"Scanned: {fname}")

            # --- PASS 2: Analyze dataset consistency & generate warnings ---
            dataset_mean_lufs = None
            if raw_lufs_values:
                dataset_mean_lufs = sum(raw_lufs_values) / len(raw_lufs_values)

            all_healthy = True
            all_reasons = []
            final_reports = {}

            for sid, report in reports.items():
                lufs = report.get("integrated_lufs")
                issues = report.get("issues", [])

                # Compare this file's LUFS against the dataset average
                if lufs is not None and dataset_mean_lufs is not None:
                    deviation = abs(lufs - dataset_mean_lufs)
                    if deviation > self.lufs_deviation_tolerance:
                        issues.append(
                            f"LUFS {lufs:.1f} deviates {deviation:.1f} LU from dataset average "
                            f"({dataset_mean_lufs:.1f}). This will skew volume consistency."
                        )
                        # Mark status as Warning if it wasn't already
                        if report["status"] != "Missing":
                            report["status"] = "Warning"

                # Update the report with the new dataset-aware warnings
                report["issues"] = issues
                final_reports[sid] = report
                all_reasons.extend(issues)
                if report["status"] == "Warning" and report["status"] != "Missing":
                    all_healthy = False

            # --- Final Summary with Recommendation ---
            summary = {
                "healthy": all_healthy,
                "total": total,
                "with_issues": sum(1 for r in final_reports.values() if r["issues"]),
                "details": final_reports,
                "reasons": all_reasons,
                # 👇 The golden recommendation for the user
                "recommended_lufs": round(dataset_mean_lufs, 1) if dataset_mean_lufs is not None else None,
                "recommendation_message": (
                    f"To ensure consistent loudness, normalize all tracks to {round(dataset_mean_lufs, 1)} LUFS "
                    f"(the average of this dataset)." if dataset_mean_lufs is not None else 
                    "Could not measure LUFS for any file. Please check ffmpeg installation."
                )
            }

            self.audit_completed.emit(summary)

        except Exception as e:
            self.error_occurred.emit(f"Audit failed: {str(e)}")

    def _get_basic_audio_info(self, path):
        """
        Returns (sample_rate, channels, duration, integrated_lufs, issues_list).
        Does NOT check against a fixed target; just measures raw values.
        """
        sr = 44100
        ch = 2
        dur = 0.0
        lufs = None
        issues = []

        # --- 1. ffprobe: fast header read ---
        try:
            cmd = [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=sample_rate,channels,duration",
                "-of", "json", path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                streams = data.get("streams", [])
                if streams:
                    s = streams[0]
                    sr = int(s.get("sample_rate", 44100))
                    ch = int(s.get("channels", 2))
                    dur = float(s.get("duration", 0.0))
            else:
                issues.append(f"ffprobe error: {result.stderr.strip()}")
        except Exception as e:
            issues.append(f"ffprobe failed: {str(e)}")

        # --- 2. ffmpeg: measure LUFS (requires decoding) ---
        if not issues or "ffprobe" not in issues[0]:
            lufs = self._measure_lufs(path)
            if lufs is None:
                issues.append("Could not measure LUFS – ffmpeg may be missing or file corrupt.")

        # --- 3. Hard technical rules (independent of dataset) ---
        if sr not in (44100, 48000):
            issues.append(f"Sample rate {sr} Hz – expected 44.1k or 48k (96k+ will skew models).")
        if ch != 2:
            issues.append(f"Expected stereo, got {ch} channel(s).")
        if 0 < dur < 10.0:
            issues.append(f"Duration {dur:.1f}s – under 10s minimum.")

        return sr, ch, dur, lufs, issues

    def _measure_lufs(self, path):
        """Extract integrated LUFS using ffmpeg's loudnorm. Returns float or None."""
        try:
            cmd = [
                "ffmpeg",
                "-i", path,
                "-af", "loudnorm=I=-14:LRA=11:TP=-1:print_format=json",
                "-f", "null",
                "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                return None

            stderr = result.stderr
            match = re.search(r'\{[^{}]*\}', stderr, re.DOTALL)
            if not match:
                return None
            data = json.loads(match.group())
            lufs_str = data.get("input_i")
            if lufs_str is None:
                return None
            return float(lufs_str)
        except Exception:
            return None

    def cancel(self):
        self.is_cancelled = True
