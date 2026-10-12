"""Central configuration shared by every stage of the pipeline.

All experiments use exactly these settings so that results across noise
conditions are directly comparable.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "speech_commands"
NOISE_DIR = DATASET_DIR / "_background_noise_"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
DATASET_NAME = "Google Speech Commands v0.02"

# --- 1. Dataset -------------------------------------------------------------
WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
SEED = 42
TEST_SPEAKER_FRACTION = 0.20   # share of speakers held out for testing
VAL_SPEAKER_FRACTION = 0.10    # share of (training) speakers used for model selection

# --- 2. Features ------------------------------------------------------------
SAMPLE_RATE = 16_000           # Hz
CLIP_SECONDS = 1.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
WIN_LENGTH = 400               # 25 ms
HOP_LENGTH = 160               # 10 ms
N_FFT = 512
WINDOW = "hann"
N_MELS = 40
F_MIN = 20.0
F_MAX = SAMPLE_RATE / 2
LOG_EPS = 1e-6
N_FRAMES = 1 + CLIP_SAMPLES // HOP_LENGTH   # 101 frames with centre padding

# --- 3. Model / training ----------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3

# --- 4. Testing -------------------------------------------------------------
# 30/20/10/0 dB as suggested in the task, extended to -5/-10 dB because SNR is
# measured over the whole clip (incl. pauses) and accuracy only collapses below 0 dB.
SNR_LEVELS_DB = [30, 20, 10, 0, -5, -10]
# Real recordings shipped with the dataset (white/pink noise files are synthetic
# and therefore excluded from the "real-world" condition).
REAL_NOISE_FILES = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "running_tap.wav",
]
