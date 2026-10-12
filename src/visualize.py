"""Figures and a Markdown results summary for the report.

Run after `evaluate`. Everything is read from `outputs/`; nothing is retrained.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import audio  # noqa: E402
from . import config as C  # noqa: E402
from .evaluate import NOISE_NAMES, RES_DIR  # noqa: E402

# palette: fixed categorical order + one-hue sequential ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq_blue", [
    "#f7fafe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "lines.linewidth": 2, "lines.markersize": 6, "legend.frameon": False,
})

SNRS = C.SNR_LEVELS_DB
MAIN_CONDS = (["clean"] + [f"white_{s}dB" for s in SNRS] + [f"real_{s}dB" for s in SNRS])
NICE = {"clean": "clean", "white": "white Gaussian", "real (mixed)": "real (mixed)"}


def label(cond):
    if cond == "clean":
        return "clean"
    kind, snr = cond.rsplit("_", 1)
    return f"{NICE.get(kind, kind.replace('_', ' '))} {snr.replace('dB', ' dB')}"


def time_axis(n):
    return np.arange(n) / C.SAMPLE_RATE


def show_logmel(ax, M, title, vmin=None, vmax=None):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=SEQ, vmin=vmin, vmax=vmax,
                   extent=[0, C.CLIP_SECONDS, 0, C.N_MELS])
    ax.set_title(title, loc="left")
    ax.set_ylabel("mel band")
    return im


# --- feature plots ----------------------------------------------------------
def fig_features(df):
    train = df[(df.split == "train") & (df.n_samples == C.CLIP_SAMPLES)]
    # detailed figure for one word
    row = train[train.word == "yes"].iloc[0]
    y = audio.fix_length(audio.load_wav(C.DATASET_DIR / row.path))
    M = audio.log_mel(y)
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.6]})
    axes[0].plot(time_axis(len(y)), y, color=SERIES[0], lw=0.8)
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f'Waveform - word "yes" ({row.path})', loc="left")
    im = show_logmel(axes[1], M, f"Log-mel spectrogram ({C.N_MELS} x {M.shape[1]})")
    axes[1].set_xlabel("time (s)")
    fig.colorbar(im, ax=axes, label="log power", shrink=0.8)
    fig.savefig(C.FIG_DIR / "feature_example_yes.png")
    plt.close(fig)

    # one example per word, shared colour scale
    fig, axes = plt.subplots(2, 5, figsize=(12, 4.4), sharex=True, sharey=True)
    mels = [audio.log_mel(audio.fix_length(audio.load_wav(
        C.DATASET_DIR / train[train.word == w].iloc[0].path))) for w in C.WORDS]
    lo, hi = np.percentile(np.stack(mels), [1, 99.9])
    for ax, w, M in zip(axes.flat, C.WORDS, mels):
        im = show_logmel(ax, M, f'"{w}"', lo, hi)
        ax.set_ylabel("")
    for ax in axes[:, 0]:
        ax.set_ylabel("mel band")
    for ax in axes[1]:
        ax.set_xlabel("time (s)")
    fig.colorbar(im, ax=axes, label="log power", shrink=0.8)
    fig.suptitle("Log-mel spectrograms, one training example per word", x=0.06, ha="left")
    fig.savefig(C.FIG_DIR / "features_all_words.png")
    plt.close(fig)


def fig_training():
    hist = pd.DataFrame(json.loads((C.OUT_DIR / "training.json").read_text())["history"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    axes[0].plot(hist.epoch, hist.train_loss, color=SERIES[0], marker="o", ms=4)
    axes[0].set(title="Training loss (cross-entropy)", xlabel="epoch", ylabel="loss")
    for i, (col, name) in enumerate([("train_acc", "train"), ("val_acc", "validation")]):
        axes[1].plot(hist.epoch, hist[col], color=SERIES[i], marker=MARKERS[i], ms=4, label=name)
    axes[1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
    axes[1].legend()
    for ax in axes:
        ax.grid(axis="y")
    fig.savefig(C.FIG_DIR / "training_curves.png")
    plt.close(fig)


# --- clean vs noisy example -------------------------------------------------
def fig_noisy_example():
    ex = np.load(RES_DIR / "example_waveforms.npz")
    conds = ["clean", "white_10dB", "white_-5dB", "real_10dB", "real_-5dB"]
    mels = {c: audio.log_mel(ex[c]) for c in conds}
    lo, hi = np.percentile(np.stack(list(mels.values())), [1, 99.9])
    ylim = np.abs(np.stack([ex[c] for c in conds])).max() * 1.05
    fig, axes = plt.subplots(len(conds), 2, figsize=(10, 9), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1.25]})
    for r, c in enumerate(conds):
        color = SERIES[0] if c == "clean" else SERIES[1] if "white" in c else SERIES[2]
        axes[r, 0].plot(time_axis(len(ex[c])), ex[c], color=color, lw=0.6)
        axes[r, 0].set_ylim(-ylim, ylim)
        axes[r, 0].set_ylabel("amplitude")
        axes[r, 0].set_title(label(c), loc="left")
        im = show_logmel(axes[r, 1], mels[c], label(c), lo, hi)
    axes[-1, 0].set_xlabel("time (s)")
    axes[-1, 1].set_xlabel("time (s)")
    fig.colorbar(im, ax=axes[:, 1], label="log power", shrink=0.6)
    fig.suptitle(f'Clean vs noisy test example - word "yes" ({ex["path"]})',
                 x=0.06, ha="left")
    fig.savefig(C.FIG_DIR / "clean_vs_noisy_example.png")
    plt.close(fig)


# --- accuracy plots ---------------------------------------------------------
def fig_accuracy(acc):
    clean = acc.loc[acc.condition == "clean", "accuracy"].item()
    x_clean = max(SNRS) + 10
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    groups = [
        (axes[0], "White Gaussian vs real-world noise", ["white", "real (mixed)"]),
        (axes[1], "Accuracy by real-world noise recording", ["white"] + NOISE_NAMES),
    ]
    for ax, title, types in groups:
        for i, t in enumerate(types):
            d = acc[acc.noise_type == t].sort_values("snr_db")
            xs, ys = list(d.snr_db) + [x_clean], list(d.accuracy) + [clean]
            ls = "--" if (t == "white" and ax is axes[1]) else "-"
            ax.plot(xs, ys, color=SERIES[i], marker=MARKERS[i], ls=ls,
                    label=NICE.get(t, t.replace("_", " ")))
        ax.axhline(1 / len(C.WORDS), color=INK2, lw=1, ls=":")
        ax.text(min(SNRS), 1 / len(C.WORDS) + 0.015, "chance (10%)", color=INK2, fontsize=8)
        ax.set_xticks(SNRS + [x_clean], [f"{s}" for s in SNRS] + ["clean"])
        ax.set(title=title, xlabel="SNR (dB)", ylim=(0, 1.02))
        ax.grid(axis="y")
        ax.legend(loc="lower right")
    axes[0].set_ylabel("test accuracy")
    fig.savefig(C.FIG_DIR / "accuracy_vs_snr.png")
    plt.close(fig)


def fig_per_word(per_word):
    data = per_word[MAIN_CONDS]
    fig, ax = plt.subplots(figsize=(12, 4.6))
    im = ax.imshow(data.values, cmap=SEQ, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MAIN_CONDS)), [label(c) for c in MAIN_CONDS], rotation=40, ha="right")
    ax.set_yticks(range(len(C.WORDS)), C.WORDS)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if v > 0.6 else INK)
    ax.set_title("Per-word accuracy (recall) by condition", loc="left")
    ax.spines[:].set_visible(False)
    fig.colorbar(im, ax=ax, label="accuracy", shrink=0.8)
    fig.savefig(C.FIG_DIR / "per_word_accuracy.png")
    plt.close(fig)


def fig_confusions(cms):
    conds = ["clean", "white_0dB", "white_-10dB", "real_0dB", "real_-10dB"]
    fig, axes = plt.subplots(1, len(conds), figsize=(20, 4.4))
    for ax, c in zip(axes, conds):
        cm = cms[c]
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        for i in range(len(C.WORDS)):
            for j in range(len(C.WORDS)):
                if norm[i, j] >= 0.02:
                    ax.text(j, i, f"{norm[i, j]:.2f}"[1:] if norm[i, j] < 1 else "1",
                            ha="center", va="center", fontsize=6.5,
                            color="white" if norm[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(C.WORDS)), C.WORDS, rotation=60)
        ax.set_yticks(range(len(C.WORDS)), C.WORDS)
        ax.set_title(f"{label(c)} (acc {np.trace(cm) / cm.sum():.2f})", loc="left")
        ax.set_xlabel("predicted")
        ax.spines[:].set_visible(False)
    axes[0].set_ylabel("true word")
    fig.suptitle("Row-normalised confusion matrices (values >= 0.02 shown)", x=0.02, ha="left")
    fig.savefig(C.FIG_DIR / "confusion_matrices.png")
    plt.close(fig)


# --- markdown summary -------------------------------------------------------
def md_table(df, floatfmt="{:.3f}"):
    df = df.copy()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(
            floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(lines)


def write_summary(acc, per_word, cms, conf):
    stats = json.loads((C.OUT_DIR / "dataset_stats.json").read_text())
    train = json.loads((C.OUT_DIR / "training.json").read_text())
    counts = pd.read_csv(C.OUT_DIR / "dataset_counts.csv")
    clean = acc.loc[acc.condition == "clean", "accuracy"].item()

    pivot = acc[acc.noise_type != "clean"].pivot(index="noise_type", columns="snr_db",
                                                 values="accuracy")
    pivot = pivot[sorted(pivot.columns, reverse=True)]
    pivot.columns = [f"{int(c)} dB" for c in pivot.columns]
    pivot.insert(0, "clean", clean)
    pivot = pivot.reindex(["white", "real (mixed)"] + NOISE_NAMES).reset_index()

    pw = per_word[MAIN_CONDS].copy()
    pw.columns = [label(c) for c in MAIN_CONDS]
    pw = pw.reset_index().rename(columns={"index": "word"})

    conf_lines = []
    for c in MAIN_CONDS:
        pairs = ", ".join(f"{t}->{p} ({k})" for t, p, k in conf[c][:5])
        conf_lines.append(f"| {label(c)} | {pairs} |")

    pred = pd.read_csv(RES_DIR / "predicted_class_counts.csv", index_col=0)[MAIN_CONDS]
    pred.columns = [label(c) for c in MAIN_CONDS]
    pred = pred.reset_index().rename(columns={"index": "predicted word"})

    hist = train["history"]
    text = f"""# Results summary (auto-generated by `python -m src.visualize`)

## 1. Dataset
* Source: {stats['dataset']} (Warden, 2018), {C.DATASET_URL}
* Words: {', '.join(stats['words'])}
* Sampling frequency: {stats['sample_rates_hz']} Hz
* Speakers: {stats['n_speakers']} in total; train {stats['speakers_per_split']['train']},
  validation {stats['speakers_per_split']['val']}, test {stats['speakers_per_split']['test']}
* Recordings: {stats['n_recordings']} in total; train {stats['recordings_per_split']['train']},
  validation {stats['recordings_per_split']['val']}, test {stats['recordings_per_split']['test']}
* Clip duration: min {stats['duration_s']['min']:.3f} s, max {stats['duration_s']['max']:.3f} s,
  mean {stats['duration_s']['mean']:.3f} s; {100 * stats['duration_s']['share_shorter_than_1s']:.1f} % shorter than 1 s (zero-padded)
* Split: speakers shuffled with seed {stats['split_seed']}; {int(100 * C.TEST_SPEAKER_FRACTION)} % of speakers -> test,
  {int(100 * C.VAL_SPEAKER_FRACTION)} % -> validation, rest -> train. Whole speakers are assigned, so
  speaker overlap is train&test = {stats['speaker_overlap']['train&test']}, train&val = {stats['speaker_overlap']['train&val']},
  val&test = {stats['speaker_overlap']['val&test']} (checked by assertion in `prepare_data.py`).

{md_table(counts, "{:.0f}")}

## 2. Features
Log-mel spectrogram: fs = {C.SAMPLE_RATE} Hz, clip length {C.CLIP_SECONDS:.0f} s ({C.CLIP_SAMPLES} samples, symmetric zero padding),
window {C.WIN_LENGTH} samples ({1000 * C.WIN_LENGTH / C.SAMPLE_RATE:.0f} ms, {C.WINDOW}), hop {C.HOP_LENGTH} samples
({1000 * C.HOP_LENGTH / C.SAMPLE_RATE:.0f} ms), FFT size {C.N_FFT}, {C.N_MELS} mel filters ({C.F_MIN:.0f}-{C.F_MAX:.0f} Hz),
power spectrum, natural log with epsilon {C.LOG_EPS}. Feature size per recording: {C.N_MELS} x {C.N_FRAMES}
(mel bands x frames). Per-mel-band mean/std normalisation computed on the training set.

## 3. Model and training
CNN, {train['trainable_parameters']} trainable parameters: 4 x [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2]
with 16/32/64/128 channels, global average pooling, dropout {train['dropout']}, Linear 128 -> 10 (softmax via loss).
Loss {train['loss']}, optimizer {train['optimizer']} (lr {train['learning_rate']}, weight decay {train['weight_decay']},
{train['lr_schedule']}), {train['epochs']} epochs, batch size {train['batch_size']}. Checkpoint with the best
validation accuracy kept (epoch {train['best_epoch']}, val acc {train['best_val_acc']:.4f};
final epoch train acc {hist[-1]['train_acc']:.4f}).

## 4. Test accuracy (n = {int(cms['clean'].sum())} test recordings, identical in every condition)

{md_table(pivot)}

### Per-word accuracy

{md_table(pw, "{:.2f}")}

### Most frequent confusions (true -> predicted, count)

| condition | top confusions |
|---|---|
{chr(10).join(conf_lines)}

### Predicted-class counts (shows bias towards particular words under noise)

{md_table(pred, "{:.0f}")}

## Figures (outputs/figures)
* `feature_example_yes.png`, `features_all_words.png` - feature plots
* `training_curves.png`
* `clean_vs_noisy_example.png` - waveform + log-mel, clean vs noisy
* `accuracy_vs_snr.png`, `per_word_accuracy.png`, `confusion_matrices.png`
"""
    (C.OUT_DIR / "results_summary.md").write_text(text)


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(C.OUT_DIR / "manifest.csv")
    acc = pd.read_csv(RES_DIR / "accuracy.csv")
    per_word = pd.read_csv(RES_DIR / "per_word_accuracy.csv", index_col=0)
    cms = dict(np.load(RES_DIR / "confusion_matrices.npz"))
    conf = json.loads((RES_DIR / "top_confusions.json").read_text())
    fig_features(df)
    fig_training()
    fig_noisy_example()
    fig_accuracy(acc)
    fig_per_word(per_word)
    fig_confusions(cms)
    write_summary(acc, per_word, cms, conf)
    print(f"figures -> {C.FIG_DIR}\nsummary -> {C.OUT_DIR / 'results_summary.md'}")


if __name__ == "__main__":
    main()
