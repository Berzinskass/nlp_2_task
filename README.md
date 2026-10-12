# Task 2: 10-word speech recognition under noise

A small CNN classifies 10 spoken command words from log-mel spectrograms. It is
trained once on clean speech and then tested under clean, white-Gaussian and
real background-noise conditions at matched SNRs (30, 20, 10, 0 dB, plus -5 and
-10 dB to show where recognition breaks down).

## Pipeline

| Step | Module | Output |
|---|---|---|
| 1. Dataset preparation | `src/prepare_data.py` | `outputs/manifest.csv`, `dataset_stats.json`, `dataset_counts.csv` |
| 2. Feature extraction | `src/features.py`, `src/audio.py` | `outputs/features_clean.npz` (cache, not committed) |
| 3. Model + training | `src/model.py`, `src/train.py` | `outputs/model.pt`, `training.json` |
| 4. Noisy testing | `src/evaluate.py` | `outputs/results/*` |
| Figures + tables | `src/visualize.py` | `outputs/figures/*.png`, `outputs/results_summary.md` |

All parameters live in `src/config.py`.

```bash
pip install -r requirements.txt
./run_all.sh            # or run each step: python -m src.prepare_data, ... 
python -m pytest tests  # SNR-scaling and feature-shape checks
```

`prepare_data` downloads Google Speech Commands v0.02 (~2.4 GB) into `data/` and
extracts only the 10 selected words and `_background_noise_`. A full run takes
about 20 minutes on a 4-core CPU (no GPU needed).

## Design choices

* **Data:** Google Speech Commands v0.02, words `yes no up down left right on off stop go`.
  Speaker ID = file-name prefix before `_nohash_`. Whole speakers are randomly
  assigned (seed 42) to test (20 %), validation (10 %) and train (70 %); an
  assertion checks there is no speaker overlap and that every word is in every split.
  The validation speakers are used only to pick the best epoch.
* **Features:** 1-s clips (zero-padded), 25 ms Hann window, 10 ms hop, 512-point FFT,
  40 mel filters (20 Hz-8 kHz), log power -> 40 x 101 per clip; per-band
  normalisation with training-set statistics.
* **Model:** 4 conv blocks (3x3 conv, BatchNorm, ReLU, 2x2 max-pool; 16/32/64/128
  channels), global average pooling, dropout 0.3, linear layer to 10 classes.
  Adam (lr 1e-3, cosine schedule), cross-entropy, 20 epochs, batch 64. No noise
  augmentation: the training data is left unchanged.
* **Noise:** added to the test waveforms only, before feature extraction.
  SNR = 10 log10(P_speech / P_noise), with P_speech the mean power of the
  original (un-padded) recording. Each test clip gets one fixed Gaussian sample
  and one fixed background segment, reused at every SNR, so only the level changes.
  *Real-world noise* = `doing_the_dishes`, `dude_miaowing`, `exercise_bike`,
  `running_tap` from the dataset (the synthetic white/pink files are excluded).
  `real (mixed)` picks one of the four at random per clip. Each recording is
  also evaluated separately to compare noise types.
