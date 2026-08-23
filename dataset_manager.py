#!/usr/bin/env python3
"""ACE-Step Dataset Manager.

Declarative audio processing, metadata generation, and Kaggle deployment
pipeline.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List
from modules.homogeneity 

import HomogeneityEngine
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class DatasetPipeline:

    def __init__(
        self,
        raw_dir: Path,
        processed_dir: Path,
        target_sr: int = 44100,
        target_lufs: float = -14.0,
        peak_ceiling_db: float = -1.0,
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.audio_out_dir = processed_dir / "audio"
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.max_peak = 10.0 ** (peak_ceiling_db / 20.0)

    def stage_normalize(self) -> List[Path]:
        """Stage 1: Two-pass EBU R128 loudness normalization and resampling."""
        logger.info("Starting Stage 1: Audio Normalization...")
        self.audio_out_dir.mkdir(parents=True, exist_ok=True)
        valid_extensions = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
        processed_files = []

        audio_files = [
            f
            for f in self.raw_dir.rglob("*")
            if f.suffix.lower() in valid_extensions
        ]
        if not audio_files:
            logger.warning(f"No audio files found in {self.raw_dir}")
            return []

        for src in audio_files:
            dest = self.audio_out_dir / f"{src.stem}_norm.wav"
            logger.info(f"Processing: {src.name} -> {dest.name}")

            # Pass 1: Read and measure integrated loudness
            data, sr = sf.read(str(src), always_2d=True)

            # Resample if mismatch occurs
            if sr != self.target_sr:
                # Basic linear resample fallback; torchaudio/librosa preferred for high-fidelity
                num_samples = int(len(data) * float(self.target_sr) / sr)
                data = np.apply_along_axis(
                    lambda ch: np.interp(
                        np.linspace(0, len(ch), num_samples),
                        np.arange(len(ch)),
                        ch,
                    ),
                    axis=0,
                    arr=data,
                )
                sr = self.target_sr

            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(data)

            # Pass 2: Shift gain to target LUFS with true peak limiting
            normalized = pyln.normalize.loudness(data, loudness, self.target_lufs)
            peak = np.max(np.abs(normalized))
            if peak > self.max_peak:
                normalized = normalized * (self.max_peak / peak)

            sf.write(str(dest), normalized, sr, subtype="PCM_24")
            processed_files.append(dest)

        logger.info(f"Stage 1 complete: {len(processed_files)} tracks normalized.")
        return processed_files

    def stage_generate_metadata(
        self, audio_files: List[Path], dataset_title: str
    ) -> Path:
        """Stage 2: Build training manifest and Kaggle metadata file."""
        logger.info("Starting Stage 2: Manifest & Metadata Generation...")
        manifest_path = self.processed_dir / "manifest.json"
        metadata_path = self.processed_dir / "dataset-metadata.json"

        manifest_entries = []
        for f in audio_files:
            info = sf.info(str(f))
            manifest_entries.append(
                {
                    "file_name": f.name,
                    "duration_sec": round(info.duration, 2),
                    "samplerate": info.samplerate,
                    "channels": info.channels,
                    "format": info.subtype,
                }
            )

        with open(manifest_path, "w", encoding="utf-8") as mf:
            json.dump(manifest_entries, mf, indent=2)

        # Kaggle Dataset Metadata schema
        kaggle_slug = (
            dataset_title.lower().replace(" ", "-").replace("_", "-")
        )
        kaggle_meta = {
            "title": dataset_title,
            "id": f"{kaggle_slug}",  # Prepend your username when uploading
            "licenses": [{"name": "CC0-1.0"}],
        }

        with open(metadata_path, "w", encoding="utf-8") as kf:
            json.dump(kaggle_meta, kf, indent=2)

        logger.info(
            f"Stage 2 complete: Manifest written to {manifest_path.name}"
        )
        return metadata_path

    def stage_kaggle_upload(
        self, username: str, dataset_slug: str, new_version: bool = False
    ):
        """Stage 3: Push processed dataset directory via Kaggle CLI."""
        logger.info("Starting Stage 3: Kaggle Upload...")
        metadata_file = self.processed_dir / "dataset-metadata.json"

        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        meta["id"] = f"{username}/{dataset_slug}"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        if new_version:
            cmd = [
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(self.processed_dir),
                "-m",
                "Automated dataset sync via dataset_manager.py",
                "-r",
                "tar",
            ]
        else:
            cmd = [
                "kaggle",
                "datasets",
                "create",
                "-p",
                str(self.processed_dir),
                "-r",
                "tar",
            ]

        logger.info(f"Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        logger.info("Stage 3 complete: Dataset uploaded successfully.")


def cmd_audit(args):
    engine = HomogeneityEngine(target_sr=args.sr, target_lufs=args.lufs)
    report = engine.analyze_directory(Path(args.target_dir))

    print("\n" + "=" * 60)
    print(
        f" DATASET HOMOGENEITY SCORE: {report.score} / 100 ({report.total_tracks} tracks)"
    )
    print("=" * 60)
    print(f"Sample Rates:      {report.sample_rates}")
    print(
        f"Loudness Spread:   Mean {report.lufs_mean} LUFS (± {report.lufs_std} LUFS)"
    )
    print(
        f"Dynamic Range:     Mean Crest {report.crest_mean_db} dB (± {report.crest_std_db} dB)"
    )
    print(f"Outlier Tracks:    {report.outliers_count} flagged")
    print("\nHarmonization Action Plan:")
    for step in report.action_plan:
        print(f" -> {step}")

    if report.outliers_count > 0:
        print("\nFlagged Outliers:")
        for t in report.track_details:
            if t.is_outlier:
                print(f" - {t.file_name}: {', '.join(t.outlier_reasons)}")
    print("")

def main():
    parser = argparse.ArgumentParser(
        description="ACE-Step Dataset Management Pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("./raw_audio"),
        help="Input raw audio folder",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("./processed_dataset"),
        help="Output dataset folder",
    )
    parser.add_argument(
        "--lufs",
        type=float,
        default=-14.0,
        help="Target integrated loudness in LUFS",
    )
    parser.add_argument(
        "--sr", type=int, default=44100, help="Target sample rate"
    )
    parser.add_argument(
        "--title",
        type=str,
        default="ACE-Step Music Dataset",
        help="Kaggle dataset title",
    )
    parser.add_argument(
        "--kaggle-user",
        type=str,
        default="",
        help="Kaggle username for upload",
    )
    parser.add_argument(
        "--kaggle-slug",
        type=str,
        default="",
        help="Kaggle dataset slug identifier",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Trigger Kaggle upload after processing",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Create a new dataset version instead of a new dataset",
    )

    args = parser.parse_args()

    pipeline = DatasetPipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.out_dir,
        target_sr=args.sr,
        target_lufs=args.lufs,
    )

    processed_files = pipeline.stage_normalize()
    if not processed_files:
        sys.exit(0)

    pipeline.stage_generate_metadata(processed_files, dataset_title=args.title)

    if args.upload:
        if not args.kaggle_user or not args.kaggle_slug:
            logger.error(
                "--kaggle-user and --kaggle-slug are required for upload."
            )
            sys.exit(1)
        pipeline.stage_kaggle_upload(
            args.kaggle_user, args.kaggle_slug, new_version=args.version
        )


if __name__ == "__main__":
    main()
