"""Step 1 - dataset preparation.

* downloads/extracts Google Speech Commands v0.02 (only the 10 selected words and
  the background-noise folder are extracted),
* builds a speaker-independent train / validation / test split,
* writes `outputs/manifest.csv` and dataset statistics for the report.

Speaker identity comes from the file name: `<speaker_hash>_nohash_<n>.wav`.
"""
import json
import tarfile
import urllib.request

import numpy as np
import pandas as pd
import soundfile as sf

from . import config as C


def download_and_extract():
    if all((C.DATASET_DIR / w).is_dir() for w in C.WORDS) and C.NOISE_DIR.is_dir():
        return
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = C.DATA_DIR / "speech_commands_v0.02.tar.gz"
    if not archive.exists():
        print(f"Downloading {C.DATASET_URL} ...")
        urllib.request.urlretrieve(C.DATASET_URL, archive)
    wanted = tuple(f"./{d}/" for d in C.WORDS + ["_background_noise_"])
    print("Extracting selected folders ...")
    with tarfile.open(archive) as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(wanted)]
        tar.extractall(C.DATASET_DIR, members=members, filter="data")


def scan():
    rows = []
    for word in C.WORDS:
        for path in sorted((C.DATASET_DIR / word).glob("*.wav")):
            info = sf.info(path)
            rows.append({
                "path": str(path.relative_to(C.DATASET_DIR)),
                "word": word,
                "label": C.WORDS.index(word),
                "speaker": path.name.split("_nohash_")[0],
                "sample_rate": info.samplerate,
                "n_samples": info.frames,
            })
    return pd.DataFrame(rows)


def speaker_split(df):
    """Assign whole speakers to test / val / train (no speaker in two sets)."""
    rng = np.random.default_rng(C.SEED)
    speakers = np.array(sorted(df["speaker"].unique()))
    rng.shuffle(speakers)
    n_test = int(round(C.TEST_SPEAKER_FRACTION * len(speakers)))
    n_val = int(round(C.VAL_SPEAKER_FRACTION * len(speakers)))
    split_of = {s: "test" for s in speakers[:n_test]}
    split_of.update({s: "val" for s in speakers[n_test:n_test + n_val]})
    split_of.update({s: "train" for s in speakers[n_test + n_val:]})
    df = df.copy()
    df["split"] = df["speaker"].map(split_of)
    return df


def check_split(df):
    sets = {s: set(df.loc[df.split == s, "speaker"]) for s in ("train", "val", "test")}
    overlaps = {
        "train&test": len(sets["train"] & sets["test"]),
        "train&val": len(sets["train"] & sets["val"]),
        "val&test": len(sets["val"] & sets["test"]),
    }
    assert not any(overlaps.values()), f"speaker overlap: {overlaps}"
    for s in ("train", "val", "test"):
        missing = set(C.WORDS) - set(df.loc[df.split == s, "word"])
        assert not missing, f"words missing from {s}: {missing}"
    return overlaps


def main():
    download_and_extract()
    df = speaker_split(scan())
    overlaps = check_split(df)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.OUT_DIR / "manifest.csv", index=False)

    per_word = df.pivot_table(index="word", columns="split", values="path",
                              aggfunc="count").reindex(C.WORDS)[["train", "val", "test"]]
    per_word["total"] = per_word.sum(axis=1)
    per_word.loc["TOTAL"] = per_word.sum()
    per_word.to_csv(C.OUT_DIR / "dataset_counts.csv")

    dur = df["n_samples"] / df["sample_rate"]
    stats = {
        "dataset": C.DATASET_NAME,
        "words": C.WORDS,
        "sample_rates_hz": sorted(df["sample_rate"].unique().tolist()),
        "n_recordings": int(len(df)),
        "n_speakers": int(df["speaker"].nunique()),
        "speakers_per_split": df.groupby("split")["speaker"].nunique().to_dict(),
        "recordings_per_split": df["split"].value_counts().to_dict(),
        "recordings_per_word": df["word"].value_counts().reindex(C.WORDS).to_dict(),
        "duration_s": {"min": float(dur.min()), "max": float(dur.max()),
                       "mean": float(dur.mean()),
                       "share_shorter_than_1s": float((dur < C.CLIP_SECONDS).mean())},
        "speaker_overlap": overlaps,
        "split_seed": C.SEED,
    }
    (C.OUT_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(per_word.to_string())


if __name__ == "__main__":
    main()
