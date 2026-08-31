"""Dataset audio embeddings + 2-D projection for visual curation.

Tiered embedding (first available wins):
  * CLAP (``laion/larger_clap_music``) if torch + transformers are installed
  * MERT (``m-a-p/MERT-v1-95M``) if available
  * a librosa mel + chroma summary vector otherwise (always works offline)

2-D reduction (first available wins):
  * UMAP (``umap-learn``)
  * t-SNE (``scikit-learn``)
  * SVD/PCA (numpy only — always works)
"""
import os

import numpy as np


def backend_label():
    """Human-readable description of the embedding + reducer in use."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        emb = "CLAP/MERT"
    except Exception:  # noqa: BLE001
        emb = "librosa (offline)"
    try:
        import umap  # noqa: F401
        red = "UMAP"
    except Exception:  # noqa: BLE001
        try:
            from sklearn.manifold import TSNE  # noqa: F401
            red = "t-SNE"
        except Exception:  # noqa: BLE001
            red = "SVD"
    return f"{emb} + {red}"


def compute_embedding(path):
    """Return a normalized embedding vector for an audio file, or ``None``."""
    if not path or not os.path.exists(path):
        return None
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        try:
            return _clap_embedding(path)
        except Exception:  # noqa: BLE001
            try:
                return _mert_embedding(path)
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    return _fingerprint_embedding(path)


_clap_cache = {"model": None, "processor": None, "device": None}


def _clap_embedding(path):
    import torch
    from transformers import ClapModel, ClapProcessor

    if _clap_cache["model"] is None:
        _clap_cache["model"] = ClapModel.from_pretrained("laion/larger_clap_music")
        _clap_cache["processor"] = ClapProcessor.from_pretrained("laion/larger_clap_music")
        _clap_cache["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _clap_cache["model"].to(_clap_cache["device"]).eval()
    model, processor, device = _clap_cache["model"], _clap_cache["processor"], _clap_cache["device"]
    inputs = processor(audios=[path], return_tensors="pt", sampling_rate=48000)
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.no_grad():
        feats = model.get_audio_features(inputs["input_features"])
    v = feats[0].cpu().numpy().astype(np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


_mert_cache = {"model": None, "processor": None, "device": None}


def _mert_embedding(path):
    import torch
    from transformers import AutoModel, AutoFeatureExtractor

    if _mert_cache["model"] is None:
        _mert_cache["model"] = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
        _mert_cache["processor"] = AutoFeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M")
        _mert_cache["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        _mert_cache["model"].to(_mert_cache["device"]).eval()
    model, processor, device = _mert_cache["model"], _mert_cache["processor"], _mert_cache["device"]
    import librosa
    y, sr = librosa.load(path, sr=24000, mono=True)
    inputs = processor(y, sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    # average the last hidden states for a compact representation
    v = out.hidden_states[-1].mean(dim=1)[0].cpu().numpy().astype(np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _fingerprint_embedding(path):
    import librosa

    y, sr = librosa.load(path, sr=16000, mono=True, duration=60)
    if len(y) == 0:
        return None
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24, hop_length=1024)
    logmel = librosa.power_to_db(mel)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=1024)
    vec = np.concatenate([
        librosa.util.fix_length(logmel.mean(axis=1), size=32),
        librosa.util.fix_length(chroma.mean(axis=1), size=16),
    ])
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def _pca_2d(X):
    Xc = X - X.mean(axis=0)
    try:
        U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :2] * S[:2]
    except Exception:  # noqa: BLE001
        return X[:, :2]


def reduce_2d(vecs):
    """Reduce a dict of {key: vector} to {key: (x, y)} for plotting."""
    items = [(int(k), v) for k, v in vecs.items() if v is not None]
    if len(items) < 2:
        return {k: (0.0, 0.0) for k, _ in items}
    keys = [k for k, _ in items]
    X = np.array([v for _, v in items], dtype=np.float64)
    try:
        import umap  # noqa: F401
        from umap import UMAP
        fit = UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.1,
                   n_components=2, random_state=42)
        Y = fit.fit_transform(X)
    except Exception:  # noqa: BLE001
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(2, len(X) - 1))
            Y = TSNE(n_components=2, random_state=42, perplexity=perp).fit_transform(X)
        except Exception:  # noqa: BLE001
            Y = _pca_2d(X)
    Y = np.asarray(Y, dtype=np.float64)
    return {k: (float(x), float(y)) for k, (x, y) in zip(keys, Y)}