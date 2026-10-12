"""Audio loading, length normalisation, noise mixing and log-mel features."""
import librosa
import numpy as np
import soundfile as sf

from . import config as C


def load_wav(path):
    """Load a mono float32 waveform at the project sampling rate."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != C.SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=C.SAMPLE_RATE)
    return y


def fix_length(y, n=C.CLIP_SAMPLES):
    """Zero-pad (symmetrically) or centre-crop the waveform to exactly n samples."""
    if len(y) >= n:
        start = (len(y) - n) // 2
        return y[start:start + n]
    pad = n - len(y)
    return np.pad(y, (pad // 2, pad - pad // 2))


def signal_power(y):
    return float(np.mean(y.astype(np.float64) ** 2))


def scale_noise_to_snr(clean, noise, snr_db, clean_power=None):
    """Return `noise` scaled so that 10*log10(P_clean / P_noise) == snr_db."""
    p_clean = signal_power(clean) if clean_power is None else clean_power
    p_noise = signal_power(noise)
    if p_noise == 0 or p_clean == 0:
        return np.zeros_like(clean)
    target = p_clean / (10 ** (snr_db / 10))
    return (noise * np.sqrt(target / p_noise)).astype(np.float32)


def add_white_noise(clean, snr_db, gaussian, clean_power=None):
    """Add white Gaussian noise; `gaussian` is a pre-drawn N(0, 1) vector."""
    return clean + scale_noise_to_snr(clean, gaussian, snr_db, clean_power)


def add_background_noise(clean, noise_track, snr_db, offset, clean_power=None):
    """Mix a segment of a long background recording (starting at `offset`)."""
    seg = noise_track[offset:offset + len(clean)]
    return clean + scale_noise_to_snr(clean, seg, snr_db, clean_power)


MEL_BASIS = librosa.filters.mel(sr=C.SAMPLE_RATE, n_fft=C.N_FFT, n_mels=C.N_MELS,
                                fmin=C.F_MIN, fmax=C.F_MAX)  # (N_MELS, N_FFT//2+1)


def log_mel(y):
    """Log-mel spectrogram of a waveform (..., samples) -> (..., N_MELS, N_FRAMES).

    Power STFT (Hann window, centred frames) -> triangular mel filterbank -> natural log.
    Accepts a single clip or a batch of equal-length clips.
    """
    spec = np.abs(librosa.stft(y, n_fft=C.N_FFT, win_length=C.WIN_LENGTH,
                               hop_length=C.HOP_LENGTH, window=C.WINDOW,
                               center=True)) ** 2
    mel = np.einsum("mf,...ft->...mt", MEL_BASIS, spec)
    return np.log(mel + C.LOG_EPS).astype(np.float32)
