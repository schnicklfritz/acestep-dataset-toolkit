import os
import json
import time
import tempfile
import shutil
import requests
from pathlib import Path
import librosa
import soundfile as sf
import numpy as np

# ----------------------------------------------------------------------
# StemSeparator – Unified stem separation with MVSEP API
# Uses mvsep.com/api (main endpoint)
# ----------------------------------------------------------------------
class StemSeparator:
    def __init__(self, config, progress_callback=None):
        """
        config: dict with keys: mvsep_api_key, kaggle_user, kaggle_key, custom_url, etc.
        progress_callback: function(percent, message)
        """
        self.config = config
        self.progress = progress_callback or (lambda p, m: None)
        self.api_token = config.get("mvsep_api_key", "").strip()
        self.base_url = "https://mvsep.com/api"

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def separate(self, audio_path, method='polarformer+multi+instrument', output_dir=None, options=None):
        """
        Main separation method.
        method: 'polarformer' | 'polarformer+multi' | 'polarformer+multi+instrument'
        options: dict with:
            - multi_model: 'roformer' (default) or 'demucs'
            - instrument_models: list of MVSEP model names
            - use_caption_recommendation: bool
            - caption_text: str (pre‑computed caption)
        Returns dict mapping stem type -> file path.
        """
        self.progress(5, f"Starting stem separation for {os.path.basename(audio_path)}...")
        output_dir = output_dir or os.path.join(os.path.dirname(audio_path), "stems")
        os.makedirs(output_dir, exist_ok=True)
        options = options or {}
        all_stems = {}

        # ---- Stage 1: PolarFormer (always run) ----
        self.progress(10, "Stage 1: Extracting vocals and instrumental via PolarFormer...")
        polar_stems = self._polarformer_separate(audio_path, output_dir)
        all_stems['vocals'] = polar_stems['vocals']
        all_stems['instrumental'] = polar_stems['instrumental']

        # ---- Stage 2: Multi‑stem separation (if requested) ----
        if 'multi' in method:
            self.progress(30, "Stage 2: Separating instrumental into multi‑stems...")
            multi_model = options.get('multi_model', 'roformer')
            multi_stems = self._multi_stem_separate(polar_stems['instrumental'], output_dir, model=multi_model)
            all_stems.update(multi_stems)

        # ---- Stage 3: Instrument‑specific MVSEP models (if requested) ----
        if 'instrument' in method:
            self.progress(60, "Stage 3: Running instrument‑specific MVSEP models...")
            instrument_models = options.get('instrument_models', [])
            if options.get('use_caption_recommendation', False):
                caption_text = options.get('caption_text', '')
                if caption_text:
                    recommended = self._recommend_models(caption_text)
                    instrument_models = list(set(instrument_models + recommended))
            if instrument_models:
                custom_stems = self._instrument_specific_separate(audio_path, output_dir, instrument_models)
                all_stems.update(custom_stems)

        self.progress(100, "Stem separation complete.")
        return all_stems

    # ------------------------------------------------------------------
    # Stage 1: PolarFormer via MVSEP API
    # ------------------------------------------------------------------
    def _polarformer_separate(self, audio_path, output_dir):
        """
        Run BS‑PolarFormer (124 bands) via MVSEP API.
        Returns dict with 'vocals' and 'instrumental' paths.
        """
        if not self.api_token:
            raise ValueError("MVSEP API key not set. Please add it in Settings.")

        model_id = 123  # BS PolarFormer (124 bands)
        result = self._call_mvsep_api(audio_path, model_id, output_dir)
        return result

    # ------------------------------------------------------------------
    # Stage 2: Multi‑stem separation via MVSEP API
    # ------------------------------------------------------------------
    def _multi_stem_separate(self, instrumental_path, output_dir, model='roformer'):
        """
        Run BS‑RoFormerSW or HTDemucs on the instrumental stem.
        Returns dict of stem paths.
        """
        if not self.api_token:
            raise ValueError("MVSEP API key not set.")

        if model == 'roformer':
            model_id = 63  # BS RoFormer SW (6 stems)
        else:  # demucs
            model_id = 20  # Demucs4 HT (4 stems)

        result = self._call_mvsep_api(instrumental_path, model_id, output_dir)
        return result

    # ------------------------------------------------------------------
    # Stage 3: Instrument‑specific models via MVSEP API
    # ------------------------------------------------------------------
    def _instrument_specific_separate(self, audio_path, output_dir, model_names):
        """
        Run each specified MVSEP model on the original mix.
        Returns dict mapping instrument name -> file path.
        """
        if not self.api_token:
            raise ValueError("MVSEP API key not set.")

        custom_stems = {}
        model_id_map = self._model_name_to_id()
        for model_name in model_names:
            model_id = model_id_map.get(model_name)
            if model_id is None:
                self.progress(65, f"Warning: No MVSEP ID found for {model_name}, skipping")
                continue
            self.progress(65, f"Running {model_name}...")
            result = self._call_mvsep_api(audio_path, model_id, output_dir)
            # result is dict mapping stem type -> file path
            # For instrument-specific models, we rename the 'target' stem
            safe_name = model_name.replace(' ', '_').lower()
            target_path = result.get('target')
            if target_path:
                new_path = os.path.join(output_dir, f"{Path(audio_path).stem}_{safe_name}.wav")
                shutil.move(target_path, new_path)
                custom_stems[safe_name] = new_path
            else:
                # If no 'target', take the first stem
                for key, path in result.items():
                    if key not in ['other']:
                        new_path = os.path.join(output_dir, f"{Path(audio_path).stem}_{safe_name}.wav")
                        shutil.move(path, new_path)
                        custom_stems[safe_name] = new_path
                        break
        return custom_stems

    # ------------------------------------------------------------------
    # MVSEP API helper – create, poll, download (CORRECTED)
    # ------------------------------------------------------------------
    def _call_mvsep_api(self, audio_path, model_id, output_dir):
        """
        Upload audio to MVSEP, wait for completion, download stems.
        Returns dict mapping stem type -> file path.
        Based on the official MVSEP API documentation.
        """
        # 1. Create separation task
        create_url = f"{self.base_url}/separation/create"
        files = {'audiofile': open(audio_path, 'rb')}
        data = {
            'api_token': self.api_token,
            'sep_type': str(model_id),
            'output_format': '1',  # 1 = WAV (16-bit)
        }

        self.progress(40, f"Uploading to MVSEP (model {model_id})...")
        resp = requests.post(create_url, data=data, files=files)
        files['audiofile'].close()
        resp.raise_for_status()

        result = resp.json()
        if not result.get('success'):
            raise RuntimeError(f"MVSEP creation failed: {result}")

        # CORRECTED: hash is inside 'data'
        job_hash = result['data']['hash']
        if not job_hash:
            raise RuntimeError(f"MVSEP returned no hash: {result}")

        self.progress(45, f"MVSEP job {job_hash} queued...")

        # 2. Poll for completion
        status_url = f"{self.base_url}/separation/get"
        poll_interval = 5
        max_wait = 600  # 10 minutes
        elapsed = 0

        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval

            status_params = {
                'api_token': self.api_token,
                'hash': job_hash
            }
            resp = requests.get(status_url, params=status_params)
            resp.raise_for_status()
            status_data = resp.json()

            if not status_data.get('success'):
                raise RuntimeError(f"MVSEP status check failed: {status_data}")

            status = status_data.get('status')

            if status == 'done':
                break
            elif status in ['waiting', 'processing', 'distributing', 'merging']:
                self.progress(45 + int(10 * elapsed / max_wait), f"Processing... {elapsed}s")
                continue
            elif status == 'failed':
                raise RuntimeError(f"MVSEP job failed: {status_data}")
            else:
                continue
        else:
            raise TimeoutError("MVSEP job timed out after 10 minutes")

        # 3. Download results
        self.progress(90, "Downloading results...")
        file_items = status_data.get('data', {}).get('files', [])
        if not file_items:
            raise RuntimeError("MVSEP returned no files")

        stem_paths = {}
        for item in file_items:
            file_url = item.get('url')
            if not file_url:
                continue
            # Get the stem type from the 'type' field
            stem_type = self._infer_stem_type_from_item(item, model_id)
            filename = item.get('download') or os.path.basename(file_url.split('?')[0])
            local_path = os.path.join(output_dir, filename)

            # Download the file
            dl_resp = requests.get(file_url, stream=True)
            dl_resp.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            stem_paths[stem_type] = local_path

        return stem_paths

    def _infer_stem_type_from_item(self, item, model_id):
        """
        Infer stem type from the 'type' field in the MVSEP response.
        """
        stem_type = item.get('type', '').lower()
        # Multi-stem models (RoFormerSW = 63, Demucs = 20)
        if model_id in [63, 20]:
            if 'vocals' in stem_type:
                return 'vocals'
            elif 'bass' in stem_type:
                return 'bass'
            elif 'drums' in stem_type:
                return 'drums'
            elif 'guitar' in stem_type:
                return 'guitar'
            elif 'piano' in stem_type:
                return 'piano'
            elif 'other' in stem_type:
                return 'other'
            else:
                return stem_type.replace(' ', '_')
        # Instrument-specific models (2 stems: target + other)
        else:
            if 'vocals' in stem_type or 'target' in stem_type:
                return 'target'
            else:
                return 'other'

    # ------------------------------------------------------------------
    # Model name to ID mapping (from MVSEP algorithms page)
    # ------------------------------------------------------------------
    def _model_name_to_id(self):
        return {
            # Ensembles
            'Ensemble (vocals, instrum)': 26,
            'Ensemble All-In (vocals, bass, drums, piano, guitar, lead/back vocals, other)': 30,
            # Multi-stem
            'BS Roformer SW (vocals, bass, drums, guitar, piano, other)': 63,
            'Demucs4 HT (vocals, drums, bass, other)': 20,
            # Keys
            'MVSep Piano': 52,
            'MVSep Organ': 58,
            'MVSep Harpsichord': 91,
            'MVSep Accordion': 99,
            'MVSep Vibraphone': 129,
            'MVSep Rhodes': 131,
            # Guitars
            'MVSep Guitar': 31,
            'MVSep Acoustic Guitar': 66,
            'MVSep Electric Guitar': 81,
            'MVSep Lead/Rhythm Guitar': 101,
            'MVSep Pedal Steel Guitar': 124,
            # Plucked Strings
            'MVSep Harp': 72,
            'MVSep Mandolin': 74,
            'MVSep Banjo': 83,
            'MVSep Sitar': 90,
            'MVSep Ukulele': 96,
            'MVSep Dobro': 97,
            # Bowed Strings
            'MVSep Violin': 65,
            'MVSep Viola': 69,
            'MVSep Cello': 70,
            'MVSep Double Bass': 73,
            # Wind
            'MVSep Saxophone': 61,
            'MVSep Flute': 67,
            'MVSep Trumpet': 71,
            'MVSep Trombone': 75,
            'MVSep Oboe': 77,
            'MVSep Clarinet': 78,
            'MVSep French Horn': 82,
            'MVSep Harmonica': 87,
            'MVSep Tuba': 92,
            'MVSep Bassoon': 93,
            'MVSep Bagpipes': 116,
            'MVSep Whistle': 132,
            'MVSep Brass': 107,
            'MVSep Woodwind': 108,
            # Bass / Drums / Synth
            'MVSep Drums': 44,
            'MVSep Bass': 41,
            'MVSep Synth': 88,
            # Percussion
            'MVSep Percussion': 105,
            'MVSep Tambourine': 76,
            'MVSep Marimba': 84,
            'MVSep Glockenspiel': 85,
            'MVSep Timpani': 86,
            'MVSep Triangle': 89,
            'MVSep Congas': 94,
            'MVSep Bells': 95,
            'MVSep Wind Chimes': 98,
            'MVSep Xylophone': 109,
            'MVSep Celesta': 110,
            'MVSep Clap': 133,
            'MVSep Cowbell': 128,
            # Vocals
            'MVSep Choir': 111,
            'MVSep Crowd removal': 34,
        }

    # ------------------------------------------------------------------
    # Recommendation engine: parse caption and map to MVSEP models
    # ------------------------------------------------------------------
    def _recommend_models(self, caption_text):
        caption_lower = caption_text.lower()
        recommended = []
        instrument_map = self._instrument_to_model_map()
        for keyword, model_name in instrument_map.items():
            if keyword in caption_lower:
                recommended.append(model_name)
        return list(set(recommended))

    def _instrument_to_model_map(self):
        return {
            'organ': 'MVSep Organ',
            'harpsichord': 'MVSep Harpsichord',
            'accordion': 'MVSep Accordion',
            'vibraphone': 'MVSep Vibraphone',
            'rhodes': 'MVSep Rhodes',
            'piano': 'MVSep Piano',
            'guitar': 'MVSep Guitar',
            'acoustic guitar': 'MVSep Acoustic Guitar',
            'electric guitar': 'MVSep Electric Guitar',
            'pedal steel guitar': 'MVSep Pedal Steel Guitar',
            'steel guitar': 'MVSep Pedal Steel Guitar',
            'harp': 'MVSep Harp',
            'mandolin': 'MVSep Mandolin',
            'banjo': 'MVSep Banjo',
            'sitar': 'MVSep Sitar',
            'ukulele': 'MVSep Ukulele',
            'dobro': 'MVSep Dobro',
            'violin': 'MVSep Violin',
            'viola': 'MVSep Viola',
            'cello': 'MVSep Cello',
            'double bass': 'MVSep Double Bass',
            'strings': 'MVSep Bowed Strings',
            'saxophone': 'MVSep Saxophone',
            'flute': 'MVSep Flute',
            'trumpet': 'MVSep Trumpet',
            'trombone': 'MVSep Trombone',
            'oboe': 'MVSep Oboe',
            'clarinet': 'MVSep Clarinet',
            'french horn': 'MVSep French Horn',
            'harmonica': 'MVSep Harmonica',
            'tuba': 'MVSep Tuba',
            'bassoon': 'MVSep Bassoon',
            'bagpipes': 'MVSep Bagpipes',
            'whistle': 'MVSep Whistle',
            'brass': 'MVSep Brass',
            'woodwind': 'MVSep Woodwind',
            'drums': 'MVSep Drums',
            'bass': 'MVSep Bass',
            'synth': 'MVSep Synth',
            'percussion': 'MVSep Percussion',
            'tambourine': 'MVSep Tambourine',
            'marimba': 'MVSep Marimba',
            'glockenspiel': 'MVSep Glockenspiel',
            'timpani': 'MVSep Timpani',
            'triangle': 'MVSep Triangle',
            'congas': 'MVSep Congas',
            'bells': 'MVSep Bells',
            'wind chimes': 'MVSep Wind Chimes',
            'xylophone': 'MVSep Xylophone',
            'celesta': 'MVSep Celesta',
            'clap': 'MVSep Clap',
            'cowbell': 'MVSep Cowbell',
            'choir': 'MVSep Choir',
            'crowd': 'MVSep Crowd removal',
            'pedal steel': 'MVSep Pedal Steel Guitar',
        }
