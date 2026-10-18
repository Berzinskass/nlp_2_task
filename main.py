"""
Task 2: 10-word speech recognition under noise.

1. Load part of the Google Speech Commands dataset and split it by speaker.
2. Turn every recording into a log-mel spectrogram.
3. Train a small CNN on the clean training recordings.
4. Test the CNN on clean speech, white noise and real background noise.

Run:  python main.py
"""
import glob
import os
import random
import tarfile
import urllib.request

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

WORDS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
DATA_DIR = "data/speech_commands"
URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
SR = 16000             # sampling frequency (Hz)
CLIPS_PER_WORD = 500   # use only part of the dataset so everything runs quickly
SNRS = [30, 20, 10, 0]
EPOCHS = 15

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------- 1. Dataset
if not os.path.isdir(DATA_DIR):
    print("Downloading the dataset (2.4 GB)...")
    os.makedirs(DATA_DIR)
    urllib.request.urlretrieve(URL, "data/speech_commands.tar.gz")
    with tarfile.open("data/speech_commands.tar.gz") as tar:
        tar.extractall(DATA_DIR)

files, labels = [], []
for label, word in enumerate(WORDS):
    word_files = sorted(glob.glob(f"{DATA_DIR}/{word}/*.wav"))
    files += random.sample(word_files, CLIPS_PER_WORD)
    labels += [label] * CLIPS_PER_WORD
labels = np.array(labels)

# The speaker ID is the part of the file name before "_nohash_".
speakers = [os.path.basename(f).split("_nohash_")[0] for f in files]
all_speakers = sorted(set(speakers))
random.shuffle(all_speakers)
test_speakers = set(all_speakers[: len(all_speakers) // 5])  # 20 % of the speakers
is_test = np.array([s in test_speakers for s in speakers])
train_speakers = set(speakers) - test_speakers

print(f"Recordings: {len(files)}, speakers: {len(all_speakers)}")
print(f"Train: {(~is_test).sum()} recordings, {len(train_speakers)} speakers")
print(f"Test:  {is_test.sum()} recordings, {len(test_speakers)} speakers")
print("Speakers in both sets:", len(train_speakers & test_speakers))
for label, word in enumerate(WORDS):
    n_train = np.sum((labels == label) & ~is_test)
    n_test = np.sum((labels == label) & is_test)
    print(f"  {word:6s} train {n_train:4d}  test {n_test:4d}")


# ------------------------------------------------------- 2. Feature extraction
def load(path):
    signal, _ = librosa.load(path, sr=SR)
    return librosa.util.fix_length(signal, size=SR)  # pad with zeros to exactly 1 s


def log_mel(signal):
    # 25 ms Hann window (400 samples), 10 ms hop (160 samples), 512-point FFT, 40 mel filters
    mel = librosa.feature.melspectrogram(y=signal, sr=SR, n_fft=512, win_length=400,
                                         hop_length=160, n_mels=40)
    return librosa.power_to_db(mel)  # shape (40, 101)


print("\nExtracting features...")
signals = [load(f) for f in files]
X = np.array([log_mel(s) for s in signals])
print("Feature shape of one recording:", X[0].shape)

# Feature plot for one word
yes_signal = signals[WORDS.index("yes") * CLIPS_PER_WORD]
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
ax[0].plot(np.arange(SR) / SR, yes_signal)
ax[0].set(title='Waveform of "yes"', xlabel="time (s)")
librosa.display.specshow(log_mel(yes_signal), sr=SR, hop_length=160, x_axis="time",
                         y_axis="mel", ax=ax[1])
ax[1].set(title='Log-mel spectrogram of "yes"')
plt.tight_layout()
plt.savefig("results/feature_yes.png")
plt.close()


# ----------------------------------------------------------------- 3. Model
mean, std = X[~is_test].mean(), X[~is_test].std()


def to_tensor(features):
    return torch.tensor((np.array(features) - mean) / std, dtype=torch.float32).unsqueeze(1)


model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # -> 16 x 20 x 50
    nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # -> 32 x 10 x 25
    nn.Flatten(),
    nn.Linear(32 * 10 * 25, 64), nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, len(WORDS)),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train, y_train = to_tensor(X[~is_test]), torch.tensor(labels[~is_test])
print("\nTraining...")
for epoch in range(EPOCHS):
    model.train()
    order = torch.randperm(len(X_train))
    losses = []
    for i in range(0, len(X_train), 32):
        batch = order[i:i + 32]
        loss = loss_fn(model(X_train[batch]), y_train[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    print(f"epoch {epoch + 1:2d}  average loss {np.mean(losses):.3f}")


def predict(features):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(features)).argmax(dim=1).numpy()


# ---------------------------------------------------------------- 4. Testing
test_signals = [s for s, t in zip(signals, is_test) if t]
y_test = labels[is_test]

real_noise_files = ["doing_the_dishes", "dude_miaowing", "exercise_bike", "running_tap"]
real_noises = [librosa.load(f"{DATA_DIR}/_background_noise_/{name}.wav", sr=SR)[0]
               for name in real_noise_files]


def white_noise(length):
    return np.random.randn(length)


def real_noise(length):
    noise = random.choice(real_noises)
    start = random.randint(0, len(noise) - length)
    return noise[start:start + length]


def add_noise(signal, noise, snr_db):
    # scale the noise so that 10*log10(signal power / noise power) = snr_db
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    scale = np.sqrt(signal_power / (noise_power * 10 ** (snr_db / 10)))
    return signal + scale * noise


def evaluate(noisy_signals):
    predictions = predict([log_mel(s) for s in noisy_signals])
    return np.mean(predictions == y_test), predictions


clean_acc, clean_pred = evaluate(test_signals)
print(f"\nClean accuracy: {clean_acc:.3f}")
accuracy = {"white": [], "real": []}
predictions = {"clean": clean_pred}
for snr in SNRS:
    for name, make_noise in [("white", white_noise), ("real", real_noise)]:
        noisy = [add_noise(s, make_noise(len(s)), snr) for s in test_signals]
        acc, pred = evaluate(noisy)
        accuracy[name].append(acc)
        predictions[f"{name} {snr} dB"] = pred
        print(f"{name:5s} noise {snr:3d} dB: accuracy {acc:.3f}")

# Accuracy for every word
print("\nAccuracy per word:")
conditions = ["clean", "white 10 dB", "real 10 dB", "white 0 dB", "real 0 dB"]
print("word    " + "".join(f"{c:>13s}" for c in conditions))
for label, word in enumerate(WORDS):
    row = [np.mean(predictions[c][y_test == label] == label) for c in conditions]
    print(f"{word:8s}" + "".join(f"{a:13.2f}" for a in row))

# Most frequent confusions
for c in ["clean", "white 0 dB", "real 0 dB"]:
    cm = confusion_matrix(y_test, predictions[c])
    np.fill_diagonal(cm, 0)
    top = np.argsort(cm, axis=None)[::-1][:5]
    pairs = [f"{WORDS[i // 10]}->{WORDS[i % 10]} ({cm.flat[i]})" for i in top]
    print(f"Top confusions, {c}: " + ", ".join(pairs))

# Plot: clean vs noisy example
example = test_signals[0]
examples = {"clean": example,
            "white noise 0 dB": add_noise(example, white_noise(SR), 0),
            "real noise 0 dB": add_noise(example, real_noise(SR), 0)}
fig, ax = plt.subplots(2, 3, figsize=(14, 5.5))
for col, (title, signal) in enumerate(examples.items()):
    ax[0, col].plot(np.arange(SR) / SR, signal)
    ax[0, col].set(title=title, xlabel="time (s)")
    librosa.display.specshow(log_mel(signal), sr=SR, hop_length=160, x_axis="time",
                             y_axis="mel", ax=ax[1, col])
plt.tight_layout()
plt.savefig("results/clean_vs_noisy.png")
plt.close()

# Plot: accuracy vs SNR
plt.figure(figsize=(8, 3.5))
plt.plot(SNRS, accuracy["white"], "o-", label="white noise")
plt.plot(SNRS, accuracy["real"], "s-", label="real noise")
plt.axhline(clean_acc, color="gray", linestyle="--", label="clean")
plt.xlabel("SNR (dB)")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("results/accuracy_vs_snr.png")
plt.close()

# Plot: confusion matrices
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for i, c in enumerate(["clean", "white 0 dB", "real 0 dB"]):
    ConfusionMatrixDisplay.from_predictions(y_test, predictions[c], display_labels=WORDS,
                                            ax=ax[i], colorbar=False)
    ax[i].set_title(c)
plt.tight_layout()
plt.savefig("results/confusion_matrices.png")
plt.close()
print("\nFigures saved in results/")
