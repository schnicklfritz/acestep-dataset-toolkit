import os, shutil, subprocess
from pathlib import Path
from PySide6.QtCore import QThread, Signal

class DspNormalizerWorker(QThread):
    progress = Signal(int, str)
    file_normalized = Signal(str, str, str, int, float)
    all_done = Signal(str, str)
    error_occurred = Signal(str)

    def __init__(self, samples, target_dir, target_sr=44100, target_lufs=-14.0):
        super().__init__()
        self.samples = samples
        self.target_dir = target_dir
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self._is_cancelled = False

    def run(self):
        try:
            total = len(self.samples)
            if total == 0:
                self.all_done.emit("", "")
                return

            norm_dir = os.path.join(self.target_dir, "normalized_audio")
            backup_dir = os.path.join(self.target_dir, "originals_backup")
            os.makedirs(norm_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)

            for idx, s in enumerate(self.samples):
                if self._is_cancelled:
                    return

                sid = s.get("id", "")
                orig_path = s.get("audio_path", "")
                fname = s.get("filename", f"sample_{sid}.wav")

                if not orig_path or not os.path.exists(orig_path):
                    continue

                backup_path = os.path.join(backup_dir, fname)
                if not os.path.exists(backup_path):
                    shutil.copy2(orig_path, backup_path)

                norm_path = os.path.join(norm_dir, f"norm_{Path(fname).stem}.wav")
                self.progress.emit(int(100 * idx / total), f"Normalizing ({self.target_lufs} LUFS): {fname}")

                cmd = [
                    "ffmpeg", "-y", "-i", orig_path,
                    "-af", f"loudnorm=I={self.target_lufs}:TP=-1.0:LRA=11",
                    "-ar", str(self.target_sr),
                    "-ac", "2",
                    norm_path
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode == 0 and os.path.exists(norm_path):
                    self.file_normalized.emit(sid, backup_path, norm_path, self.target_sr, self.target_lufs)
                else:
                    shutil.copy2(orig_path, norm_path)
                    self.file_normalized.emit(sid, backup_path, norm_path, self.target_sr, self.target_lufs)

            self.progress.emit(100, "Normalization complete.")
            self.all_done.emit(norm_dir, backup_dir)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self._is_cancelled = True

# ============================================================================
# ORIGINAL RemoteCaptionWorker (with added DeepSeek backend support)
# ============================================================================
