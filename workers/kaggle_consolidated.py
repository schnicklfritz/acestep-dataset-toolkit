# workers/kaggle_consolidated.py
# ⚙️ Independent Background Thread Poller: Keeps UI completely fluid during GPU runs

import json
import os
import subprocess
import time
from PySide6.QtCore import QThread, Signal


class KaggleConsolidatedWorker(QThread):
    progress = Signal(int, str)
    all_done = Signal(dict)
    failed = Signal(str)

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.is_cancelled = False

    def run(self):
        try:
            username = self.config.get("kaggle_user", "")
            # This must exactly match the notebook slug name configured on your Kaggle dashboard
            notebook_slug = "ace-step-master-pipeline"
            kernel_id = f"{username}/{notebook_slug}"

            if not username:
                self.failed.emit("Configuration error: Kaggle username is blank inside Settings.")
                return

            self.progress.emit(40, "Remote cluster engaged. Waiting for Kaggle container initialization...")
            time.sleep(5)  # Give the API a brief moment to register the initial kernel push command

            start_time = time.time()
            
            # --- ASYNCHRONOUS POLLING MATRIX ---
            while not self.is_cancelled:
                # Execute the standard low-overhead CLI query to read remote container logs
                cmd = ["kaggle", "kernels", "status", kernel_id]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                output = res.stdout.lower()

                # Calculate run time to display a live clock inside your status panel
                elapsed_min = int((time.time() - start_time) // 60)
                elapsed_sec = int((time.time() - start_time) % 60)
                timer_str = f"[{elapsed_min:02d}:{elapsed_sec:02d}]"

                if "running" in output:
                    self.progress.emit(60, f"🚀 Dual T4 GPUs Active {timer_str} — Crunching audio, BPM, and WhisperX...")
                elif "complete" in output or "has status 'complete'" in output:
                    self.progress.emit(85, "✨ Remote processing complete! Fetching manifest files...")
                    break
                elif "error" in output or "failed" in output:
                    self.failed.emit(f"Remote processing exception {timer_str}: Kaggle environment threw an internal runtime compilation error.")
                    return
                elif "queued" in output:
                    self.progress.emit(50, "⏳ Cloud container queued. Waiting for dual-T4 cluster allocation slot...")
                else:
                    # Fallback check if the local environment configuration tool returns unexpected syntax
                    if res.returncode != 0:
                        self.failed.emit(f"Connection anomaly: {res.stderr.strip() or 'Is kaggle.json missing from your computer?'}")
                        return
                    self.progress.emit(45, f"Connecting {timer_str} — Synchronizing pipeline variables...")

                # Sleep for 15 seconds to minimize network noise and protect your local loop cycles
                time.sleep(15)

            if self.is_cancelled:
                return

            # --- ARTIFACT RETRIEVAL STAGE ---
            # Download the final compiled artifact (audit_results.json) straight to your local directory
            download_cmd = ["kaggle", "kernels", "output", kernel_id, "-p", os.path.dirname(os.path.dirname(__file__))]
            download_res = subprocess.run(download_cmd, capture_output=True, text=True, check=False)

            if download_res.returncode != 0:
                self.failed.emit(f"Artifact retrieval failure: Could not download audit_results.json from cloud storage. {download_res.stderr}")
                return

            # Emit final success token back to DatasetManager to trigger the local layout sync
            self.all_done.emit({"status": "Success"})

        except Exception as e:
            self.failed.emit(f"Local poller thread failure: {str(e)}")

    def cancel(self):
        """Standard thread abort hook."""
        self.is_cancelled = True
