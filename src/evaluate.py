"""Step 4 - evaluate the trained model under clean and noisy test conditions.

Noise is added to the *test* waveforms only, before feature extraction.
Every condition uses the same test recordings, and for each recording the
same noise realisation (same Gaussian sample / same background segment) is
reused at every SNR, so only the noise level changes between SNR levels.

SNR = 10*log10(P_speech / P_noise), where P_speech is the mean power of the
original (un-padded) recording and the noise covers the full 1-s clip.
"""
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix

from . import audio
from . import config as C
from .features import load_waveforms, log_mel_batched
from .model import KeywordCNN
from .train import MODEL_PATH, predict

RES_DIR = C.OUT_DIR / "results"
NOISE_NAMES = [f.removesuffix(".wav") for f in C.REAL_NOISE_FILES]


def load_model():
    model = KeywordCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    return model.eval()


def noise_plan(n):
    """Fixed per-utterance noise choices (shared by all SNR levels)."""
    rng = np.random.default_rng(C.SEED + 1)
    tracks = [audio.load_wav(C.NOISE_DIR / f) for f in C.REAL_NOISE_FILES]
    white = rng.standard_normal((n, C.CLIP_SAMPLES)).astype(np.float32)
    # one random start offset per (utterance, noise file)
    offsets = np.stack([rng.integers(0, len(t) - C.CLIP_SAMPLES, size=n) for t in tracks], 1)
    mixed_choice = rng.integers(0, len(tracks), size=n)
    return tracks, white, offsets, mixed_choice


def make_conditions(waves, power):
    """Yield (name, noise_type, snr, noisy_waveforms)."""
    n = len(waves)
    tracks, white, offsets, choice = noise_plan(n)
    yield "clean", "clean", None, waves
    for snr in C.SNR_LEVELS_DB:
        yield f"white_{snr}dB", "white", snr, np.stack([
            audio.add_white_noise(waves[i], snr, white[i], power[i])
            for i in range(n)])
    for snr in C.SNR_LEVELS_DB:
        yield f"real_{snr}dB", "real (mixed)", snr, np.stack([
            audio.add_background_noise(waves[i], tracks[choice[i]], snr,
                                       offsets[i, choice[i]], power[i])
            for i in range(n)])
    for k, name in enumerate(NOISE_NAMES):
        for snr in C.SNR_LEVELS_DB:
            yield f"{name}_{snr}dB", name, snr, np.stack([
                audio.add_background_noise(waves[i], tracks[k], snr, offsets[i, k], power[i])
                for i in range(n)])


def top_confusions(cm, k=8):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.dstack(np.unravel_index(np.argsort(off, axis=None)[::-1][:k], off.shape))[0]
    return [(C.WORDS[t], C.WORDS[p], int(off[t, p])) for t, p in idx if off[t, p] > 0]


def main():
    RES_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.OUT_DIR / "manifest.csv")
    test = df[df.split == "test"].reset_index(drop=True)
    y = test["label"].to_numpy()
    model = load_model()
    print(f"Loading {len(test)} test recordings ...")
    waves, power = load_waveforms(test["path"].tolist())

    rows, per_word, pred_dist, confusions, cms = [], {}, {}, {}, {}
    # example clip for the waveform / spectrogram figure: first full-length "yes"
    example_idx = int(test.index[(test.word == "yes") & (test.n_samples == C.CLIP_SAMPLES)][0])
    examples = {}
    for name, ntype, snr, noisy in make_conditions(waves, power):
        pred = predict(model, log_mel_batched(noisy))
        acc = float((pred == y).mean())
        cm = confusion_matrix(y, pred, labels=range(len(C.WORDS)))
        rows.append({"condition": name, "noise_type": ntype, "snr_db": snr, "accuracy": acc})
        per_word[name] = cm.diagonal() / cm.sum(1)
        pred_dist[name] = cm.sum(0)
        confusions[name] = top_confusions(cm)
        cms[name] = cm
        examples[name] = noisy[example_idx]
        print(f"{name:28s} acc = {acc:.4f}")
        # keep per-utterance predictions for further analysis
        test[f"pred_{name}"] = [C.WORDS[p] for p in pred]

    acc = pd.DataFrame(rows)
    acc.to_csv(RES_DIR / "accuracy.csv", index=False)
    pd.DataFrame(per_word, index=C.WORDS).to_csv(RES_DIR / "per_word_accuracy.csv")
    pd.DataFrame(pred_dist, index=C.WORDS).to_csv(RES_DIR / "predicted_class_counts.csv")
    (RES_DIR / "top_confusions.json").write_text(json.dumps(confusions, indent=1))
    np.savez(RES_DIR / "confusion_matrices.npz", **cms)
    np.savez(RES_DIR / "example_waveforms.npz", path=test["path"][example_idx], **examples)
    test.drop(columns=["split"]).to_csv(RES_DIR / "test_predictions.csv", index=False)
    print(acc.to_string(index=False))


if __name__ == "__main__":
    main()
