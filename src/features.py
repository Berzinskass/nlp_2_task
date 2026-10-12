"""Step 2 - feature extraction (log-mel spectrograms) with on-disk caching."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import audio
from . import config as C

CACHE = C.OUT_DIR / "features_clean.npz"
CHUNK = 2000


def _load(rel_path):
    """Return (fixed-length waveform, power of the original un-padded recording)."""
    y = audio.load_wav(C.DATASET_DIR / rel_path)
    return audio.fix_length(y), audio.signal_power(y)


def load_waveforms(paths):
    """Load clips as an (N, CLIP_SAMPLES) array plus per-clip speech power."""
    with ThreadPoolExecutor() as ex:
        out = list(ex.map(_load, paths))
    return np.stack([w for w, _ in out]), np.array([p for _, p in out])


def log_mel_batched(waves):
    return np.concatenate([audio.log_mel(waves[i:i + CHUNK])
                           for i in range(0, len(waves), CHUNK)])


def extract(paths):
    feats = []
    for i in range(0, len(paths), CHUNK):
        waves, _ = load_waveforms(paths[i:i + CHUNK])
        feats.append(audio.log_mel(waves))
    return np.concatenate(feats)


def load_clean_features():
    """Return (manifest, X) where X[i] is the clean log-mel of manifest row i."""
    df = pd.read_csv(C.OUT_DIR / "manifest.csv")
    if CACHE.exists():
        cached = np.load(CACHE, allow_pickle=False)
        if list(cached["paths"]) == df["path"].tolist():
            return df, cached["X"]
    print(f"Extracting log-mel features for {len(df)} recordings ...")
    X = extract(df["path"].tolist())
    np.savez(CACHE, X=X, paths=np.array(df["path"].tolist()))
    return df, X


def main():
    df, X = load_clean_features()
    print("feature tensor:", X.shape, "(recordings, mel bands, frames)")


if __name__ == "__main__":
    main()
