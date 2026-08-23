#!/usr/bin/env python3
"""ACE-Step Dataset Manager.

Unified GUI & CLI for dataset homogeneity auditing, two-pass EBU R128
loudness normalization, manifest creation, and Kaggle publishing.
"""

import argparse
import json
import logging
from pathlib import Path
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

# Import local Homogeneity Module
try:
    from modules.homogeneity import HomogeneityEngine, HomogeneityReport

    HOMOGENEITY_AVAILABLE = True
except ImportError:
    HOMOGENEITY_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("dataset_manager")


# =====================================================================
# BACKEND DSP & PIPELINE ENGINE
# =====================================================================
class AudioPipelineEngine:

    @staticmethod
    def normalize_file(
        src: Path,
        dest: Path,
        target_lufs: float = -14.0,
        target_sr: int = 44100,
        peak_ceiling_db: float = -1.0,
    ):
        """Execute 2-pass EBU R128 normalization and sample rate alignment."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        max_peak = 10.0 ** (peak_ceiling_db / 20.0)

        data, sr = sf.read(str(src), always_2d=True)

        # Resample if mismatch occurs
        if sr != target_sr:
            num_samples = int(len(data) * float(target_sr) / sr)
            data = np.apply_along_axis(
                lambda ch: np.interp(
                    np.linspace(0, len(ch), num_samples),
                    np.arange(len(ch)),
                    ch,
                ),
                axis=0,
                arr=data,
            )
            sr = target_sr

        # Pass 1: Measure loudness
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(data)

        # Pass 2: Apply linear gain shift and true peak limiter guard
        normalized = pyln.normalize.loudness(data, loudness, target_lufs)
        peak = np.max(np.abs(normalized))
        if peak > max_peak:
            normalized = normalized * (max_peak / peak)

        sf.write(str(dest), normalized, sr, subtype="PCM_24")

    @staticmethod
    def generate_manifest(dataset_dir: Path, title: str, kaggle_user: str):
        """Generate manifest.json and dataset-metadata.json."""
        audio_dir = (
            dataset_dir / "audio"
            if (dataset_dir / "audio").exists()
            else dataset_dir
        )
        files = sorted(
            [
                f
                for f in audio_dir.rglob("*")
                if f.suffix.lower() in {".wav", ".flac"}
            ]
        )

        entries = []
        for f in files:
            info = sf.info(str(f))
            entries.append(
                {
                    "file_name": f.name,
                    "duration_sec": round(info.duration, 2),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "format": info.subtype,
                }
            )

        manifest_path = dataset_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(entries, mf, indent=2)

        kaggle_slug = title.lower().replace(" ", "-").replace("_", "-")
        meta_path = dataset_dir / "dataset-metadata.json"
        with open(meta_path, "w", encoding="utf-8") as kf:
            json.dump(
                {
                    "title": title,
                    "id": (
                        f"{kaggle_user}/{kaggle_slug}"
                        if kaggle_user
                        else kaggle_slug
                    ),
                    "licenses": [{"name": "CC0-1.0"}],
                },
                kf,
                indent=2,
            )
        return manifest_path, len(entries)


# =====================================================================
# FULL DESKTOP GUI
# =====================================================================
class DatasetToolkitApp:

    def __init__(self, root):
        self.root = root
        self.root.title("ACE-Step Dataset Toolkit")
        self.root.geometry("1020x720")
        self.root.minsize(850, 580)

        self.style = ttk.Style()
        self.style.theme_use("clam")

        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_homogeneity = ttk.Frame(self.notebook)
        self.tab_normalize = ttk.Frame(self.notebook)
        self.tab_kaggle = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_homogeneity, text="1. Homogeneity Audit")
        self.notebook.add(self.tab_normalize, text="2. DSP Normalizer")
        self.notebook.add(self.tab_kaggle, text="3. Kaggle Deployment")

        self._build_homogeneity_tab()
        self._build_normalize_tab()
        self._build_kaggle_tab()

    # --- TAB 1: AUDIT ---
    def _build_homogeneity_tab(self):
        top = ttk.LabelFrame(self.tab_homogeneity, text="Audio Source Folder")
        top.pack(fill="x", padx=10, pady=5)

        self.audit_dir_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(top, textvariable=self.audit_dir_var).pack(
            side="left", fill="x", expand=True, padx=5, pady=5
        )
        ttk.Button(
            top,
            text="Browse",
            command=lambda: self._browse(self.audit_dir_var),
        ).pack(side="left", padx=5)
        self.btn_audit = ttk.Button(
            top,
            text="Analyze Homogeneity",
            command=self._run_audit_thread,
        )
        self.btn_audit.pack(side="left", padx=5)

        summary = ttk.Frame(self.tab_homogeneity)
        summary.pack(fill="x", padx=10, pady=5)

        # Score Box
        score_frame = ttk.LabelFrame(summary, text="Consistency Score")
        score_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.lbl_score = ttk.Label(
            score_frame,
            text="-- / 100",
            font=("Helvetica", 20, "bold"),
            foreground="#2B8A3E",
        )
        self.lbl_score.pack(pady=4)
        self.progress_score = ttk.Progressbar(
            score_frame, orient="horizontal", length=160, mode="determinate"
        )
        self.progress_score.pack(padx=10, pady=4)
        self.lbl_status = ttk.Label(
            score_frame, text="Status: Ready", font=("Helvetica", 9, "italic")
        )
        self.lbl_status.pack(pady=2)

        # Stats Box
        stats_frame = ttk.LabelFrame(summary, text="Acoustic Profile")
        stats_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.lbl_stat_tracks = ttk.Label(
            stats_frame, text="• Total Tracks: --"
        )
        self.lbl_stat_tracks.pack(anchor="w", padx=10, pady=2)
        self.lbl_stat_lufs = ttk.Label(
            stats_frame, text="• Loudness Spread: --"
        )
        self.lbl_stat_lufs.pack(anchor="w", padx=10, pady=2)
        self.lbl_stat_crest = ttk.Label(
            stats_frame, text="• Dynamic Range (Crest): --"
        )
        self.lbl_stat_crest.pack(anchor="w", padx=10, pady=2)
        self.lbl_stat_centroid = ttk.Label(
            stats_frame, text="• Timbre Centroid: --"
        )
        self.lbl_stat_centroid.pack(anchor="w", padx=10, pady=2)

        # Action Plan Box
        action_frame = ttk.LabelFrame(
            self.tab_homogeneity, text="Harmonization Action Plan"
        )
        action_frame.pack(fill="x", padx=10, pady=5)
        self.txt_action = tk.Text(
            action_frame, height=3, wrap="word", bg="#F8F9FA", relief="flat"
        )
        self.txt_action.pack(fill="both", expand=True, padx=5, pady=5)
        self.txt_action.insert(
            "1.0",
            "Click 'Analyze Homogeneity' to scan for sample rate clashes, loudness drift, and mastering outliers.",
        )
        self.txt_action.config(state="disabled")

        # Table
        tbl_frame = ttk.LabelFrame(
            self.tab_homogeneity, text="Track Consistency Matrix"
        )
        tbl_frame.pack(fill="both", expand=True, padx=10, pady=5)

        cols = (
            "file",
            "rate",
            "lufs",
            "peak",
            "crest",
            "centroid",
            "warnings",
        )
        self.tree_audit = ttk.Treeview(
            tbl_frame, columns=cols, show="headings"
        )
        self.tree_audit.heading("file", text="Filename")
        self.tree_audit.heading("rate", text="Rate")
        self.tree_audit.heading("lufs", text="LUFS")
        self.tree_audit.heading("peak", text="Peak dB")
        self.tree_audit.heading("crest", text="Crest (DR dB)")
        self.tree_audit.heading("centroid", text="Centroid (Hz)")
        self.tree_audit.heading("warnings", text="Audit Warnings")

        self.tree_audit.column("file", width=240)
        self.tree_audit.column("rate", width=80)
        self.tree_audit.column("lufs", width=70)
        self.tree_audit.column("peak", width=70)
        self.tree_audit.column("crest", width=90)
        self.tree_audit.column("centroid", width=90)
        self.tree_audit.column("warnings", width=250)

        self.tree_audit.tag_configure(
            "outlier", background="#FFE3E3", foreground="#C92A2A"
        )
        self.tree_audit.tag_configure(
            "normal", background="#FFFFFF", foreground="#212529"
        )

        scroll = ttk.Scrollbar(
            tbl_frame, orient="vertical", command=self.tree_audit.yview
        )
        self.tree_audit.configure(yscrollcommand=scroll.set)
        self.tree_audit.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    # --- TAB 2: NORMALIZER ---
    def _build_normalize_tab(self):
        f = ttk.LabelFrame(
            self.tab_normalize, text="Batch Two-Pass Normalization Settings"
        )
        f.pack(fill="both", expand=True, padx=15, pady=15)

        ttk.Label(f, text="Raw Input Directory:").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )
        self.norm_in_var = tk.StringVar(value=str(Path.cwd() / "raw_audio"))
        ttk.Entry(f, textvariable=self.norm_in_var, width=45).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            f, text="Browse", command=lambda: self._browse(self.norm_in_var)
        ).grid(row=0, column=2)

        ttk.Label(f, text="Processed Output Directory:").grid(
            row=1, column=0, sticky="w", padx=10, pady=8
        )
        self.norm_out_var = tk.StringVar(
            value=str(Path.cwd() / "processed_dataset")
        )
        ttk.Entry(f, textvariable=self.norm_out_var, width=45).grid(
            row=1, column=1, padx=5
        )
        ttk.Button(
            f, text="Browse", command=lambda: self._browse(self.norm_out_var)
        ).grid(row=1, column=2)

        ttk.Label(f, text="Target Integrated Loudness (LUFS):").grid(
            row=2, column=0, sticky="w", padx=10, pady=8
        )
        self.lufs_var = tk.DoubleVar(value=-14.0)
        ttk.Scale(
            f, from_=-24.0, to=-9.0, variable=self.lufs_var, length=200
        ).grid(row=2, column=1, sticky="w", padx=5)

        ttk.Label(f, text="Output Sample Rate:").grid(
            row=3, column=0, sticky="w", padx=10, pady=8
        )
        self.sr_combo = ttk.Combobox(
            f, values=["44100", "48000"], state="readonly", width=10
        )
        self.sr_combo.set("44100")
        self.sr_combo.grid(row=3, column=1, sticky="w", padx=5)

        self.btn_run_norm = ttk.Button(
            f,
            text="Execute 2-Pass Loudness Normalization",
            command=self._run_norm_thread,
        )
        self.btn_run_norm.grid(row=4, column=0, columnspan=3, pady=20)

        self.norm_log = tk.Text(
            f, height=10, wrap="word", bg="#F8F9FA", relief="solid", bd=1
        )
        self.norm_log.grid(
            row=5, column=0, columnspan=3, sticky="nsew", padx=10, pady=10
        )
        f.rowconfigure(5, weight=1)

    # --- TAB 3: KAGGLE ---
    def _build_kaggle_tab(self):
        f = ttk.LabelFrame(
            self.tab_kaggle, text="Kaggle Dataset Deployment & Manifest"
        )
        f.pack(fill="both", expand=True, padx=15, pady=15)

        ttk.Label(f, text="Processed Dataset Path:").grid(
            row=0, column=0, sticky="w", padx=10, pady=8
        )
        self.kag_dir_var = tk.StringVar(
            value=str(Path.cwd() / "processed_dataset")
        )
        ttk.Entry(f, textvariable=self.kag_dir_var, width=45).grid(
            row=0, column=1, padx=5
        )
        ttk.Button(
            f, text="Browse", command=lambda: self._browse(self.kag_dir_var)
        ).grid(row=0, column=2)

        ttk.Label(f, text="Dataset Title:").grid(
            row=1, column=0, sticky="w", padx=10, pady=8
        )
        self.title_var = tk.StringVar(value="ACE-Step Music Dataset")
        ttk.Entry(f, textvariable=self.title_var, width=45).grid(
            row=1, column=1, padx=5
        )

        ttk.Label(f, text="Kaggle Username:").grid(
            row=2, column=0, sticky="w", padx=10, pady=8
        )
        self.user_var = tk.StringVar(value="")
        ttk.Entry(f, textvariable=self.user_var, width=45).grid(
            row=2, column=1, padx=5
        )

        self.is_version_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            f,
            text="Upload as new version of existing dataset",
            variable=self.is_version_var,
        ).grid(row=3, column=1, sticky="w", pady=5)

        ttk.Button(
            f,
            text="Generate Manifest & Deploy to Kaggle",
            command=self._run_upload_thread,
        ).grid(row=4, column=0, columnspan=3, pady=20)

    # --- HELPERS & THREAD WORKERS ---
    def _browse(self, str_var):
        d = filedialog.askdirectory()
        if d:
            str_var.set(d)

    def _run_audit_thread(self):
        target = Path(self.audit_dir_var.get())
        if not target.exists():
            messagebox.showerror(
                "Error", f"Directory does not exist:\n{target}"
            )
            return

        self.btn_audit.config(state="disabled")
        self.lbl_status.config(text="Status: Auditing audio...")

        def worker():
            engine = HomogeneityEngine(target_sr=44100, target_lufs=-14.0)
            report = engine.analyze_directory(target)
            self.root.after(0, lambda: self._populate_audit(report))

        threading.Thread(target=worker, daemon=True).start()

    def _populate_audit(self, report: "HomogeneityReport"):
        self.btn_audit.config(state="normal")
        self.lbl_score.config(text=f"{report.score:.1f} / 100")
        self.progress_score["value"] = report.score

        if report.score >= 80.0:
            self.lbl_score.config(foreground="#2B8A3E")
            self.lbl_status.config(text="Status: Highly Homogeneous")
        elif report.score >= 50.0:
            self.lbl_score.config(foreground="#E67700")
            self.lbl_status.config(text="Status: Moderate Variance")
        else:
            self.lbl_score.config(foreground="#C92A2A")
            self.lbl_status.config(text="Status: Outliers Detected")

        sr_str = ", ".join(f"{s}Hz" for s in report.sample_rates)
        self.lbl_stat_tracks.config(
            text=f"• Total Tracks: {report.total_tracks} ({sr_str})"
        )
        self.lbl_stat_lufs.config(
            text=f"• Loudness: Mean {report.lufs_mean} LUFS (± {report.lufs_std} LUFS)"
        )
        self.lbl_stat_crest.config(
            text=f"• Dynamic Range: Mean Crest {report.crest_mean_db} dB (± {report.crest_std_db} dB)"
        )
        self.lbl_stat_centroid.config(
            text=f"• Timbre Centroid: Mean {report.centroid_mean_hz} Hz (± {report.centroid_std_hz} Hz)"
        )

        self.txt_action.config(state="normal")
        self.txt_action.delete("1.0", "end")
        for plan in report.action_plan:
            self.txt_action.insert("end", f"• {plan}\n")
        self.txt_action.config(state="disabled")

        for item in self.tree_audit.get_children():
            self.tree_audit.delete(item)

        for t in report.track_details:
            tag = "outlier" if t.is_outlier else "normal"
            warn = "; ".join(t.outlier_reasons) if t.is_outlier else "OK"
            self.tree_audit.insert(
                "",
                "end",
                values=(
                    t.file_name,
                    f"{t.sample_rate} Hz",
                    f"{t.lufs:.1f}",
                    f"{t.peak_db:.1f}",
                    f"{t.crest_factor_db:.1f}",
                    f"{t.spectral_centroid_hz:.0f}",
                    warn,
                ),
                tags=(tag,),
            )

    def _run_norm_thread(self):
        in_d = Path(self.norm_in_var.get())
        out_d = Path(self.norm_out_var.get())
        lufs = self.lufs_var.get()
        sr = int(self.sr_combo.get())

        if not in_d.exists():
            messagebox.showerror("Error", f"Input directory missing:\n{in_d}")
            return

        self.btn_run_norm.config(state="disabled")
        self.norm_log.delete("1.0", "end")

        def worker():
            files = sorted(
                [
                    f
                    for f in in_d.rglob("*")
                    if f.suffix.lower()
                    in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
                ]
            )
            self.norm_log.insert(
                "end",
                f"Starting batch normalization for {len(files)} tracks...\n",
            )
            for f in files:
                dest = out_d / "audio" / f"{f.stem}_norm.wav"
                AudioPipelineEngine.normalize_file(
                    f, dest, target_lufs=lufs, target_sr=sr
                )
                self.norm_log.insert("end", f"Processed: {f.name} -> {dest.name}\n")
                self.norm_log.see("end")
            self.norm_log.insert(
                "end", "\nCompleted: All files normalized and resampled.\n"
            )
            self.btn_run_norm.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()

    def _run_upload_thread(self):
        d = Path(self.kag_dir_var.get())
        title = self.title_var.get()
        user = self.user_var.get()
        is_ver = self.is_version_var.get()

        if not d.exists():
            messagebox.showerror(
                "Error", f"Processed dataset folder missing:\n{d}"
            )
            return

        try:
            AudioPipelineEngine.generate_manifest(d, title, user)
            action = "version" if is_ver else "create"
            cmd = ["kaggle", "datasets", action, "-p", str(d), "-r", "tar"]
            if is_ver:
                cmd.extend(["-m", "Dataset synced via toolkit"])

            subprocess.run(cmd, check=True)
            messagebox.showinfo(
                "Success", f"Dataset '{title}' successfully pushed to Kaggle!"
            )
        except Exception as e:
            messagebox.showerror("Upload Failed", str(e))


# =====================================================================
# CLI PARSER DISPATCHER
# =====================================================================
def main():
    if len(sys.argv) == 1 or "--gui" in sys.argv:
        root = tk.Tk()
        app = DatasetToolkitApp(root)
        root.mainloop()
    else:
        parser = argparse.ArgumentParser(
            description="ACE-Step Dataset Management Toolkit"
        )
        subparsers = parser.add_subparsers(dest="command", required=True)

        # CLI: audit
        p_aud = subparsers.add_parser(
            "audit", help="Audit dataset homogeneity"
        )
        p_aud.add_argument(
            "target_dir", nargs="?", default=".", help="Directory to scan"
        )
        p_aud.add_argument(
            "--lufs", type=float, default=-14.0, help="Target LUFS"
        )
        p_aud.add_argument(
            "--sr", type=int, default=44100, help="Target sample rate"
        )

        # CLI: normalize
        p_norm = subparsers.add_parser(
            "normalize", help="Run 2-pass loudness normalizer"
        )
        p_norm.add_argument(
            "--raw-dir", default="./raw_audio", help="Input folder"
        )
        p_norm.add_argument(
            "--out-dir", default="./processed_dataset", help="Output folder"
        )
        p_norm.add_argument(
            "--lufs", type=float, default=-14.0, help="Target LUFS"
        )
        p_norm.add_argument(
            "--sr", type=int, default=44100, help="Target sample rate"
        )

        # CLI: deploy
        p_dep = subparsers.add_parser(
            "deploy", help="Generate manifest and upload to Kaggle"
        )
        p_dep.add_argument(
            "--dataset-dir",
            default="./processed_dataset",
            help="Dataset directory",
        )
        p_dep.add_argument(
            "--title", default="ACE-Step Music Dataset", help="Dataset title"
        )
        p_dep.add_argument(
            "--kaggle-user", default="", help="Kaggle username"
        )
        p_dep.add_argument(
            "--version",
            action="store_true",
            help="Update version instead of creating",
        )

        args = parser.parse_args()

        if args.command == "audit":
            engine = HomogeneityEngine(target_sr=args.sr, target_lufs=args.lufs)
            rep = engine.analyze_directory(Path(args.target_dir))
            print(f"\nHOMOGENEITY SCORE: {rep.score}/100 ({rep.total_tracks} tracks)")
            print(f"Loudness Spread:   Mean {rep.lufs_mean} LUFS (± {rep.lufs_std} LUFS)")
            print(f"Dynamic Range:     Mean Crest {rep.crest_mean_db} dB (± {rep.crest_std_db} dB)")
            print("\nAction Plan:")
            for a in rep.action_plan:
                print(f" -> {a}")

        elif args.command == "normalize":
            in_d, out_d = Path(args.raw_dir), Path(args.out_dir)
            files = [
                f
                for f in in_d.rglob("*")
                if f.suffix.lower()
                in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
            ]
            for f in files:
                dest = out_d / "audio" / f"{f.stem}_norm.wav"
                AudioPipelineEngine.normalize_file(
                    f, dest, target_lufs=args.lufs, target_sr=args.sr
                )
                print(f"Normalized: {f.name} -> {dest.name}")

        elif args.command == "deploy":
            AudioPipelineEngine.generate_manifest(
                Path(args.dataset_dir), args.title, args.kaggle_user
            )
            act = "version" if args.version else "create"
            cmd = [
                "kaggle",
                "datasets",
                act,
                "-p",
                args.dataset_dir,
                "-r",
                "tar",
            ]
            if args.version:
                cmd.extend(["-m", "Dataset synced via toolkit"])
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
