import numpy as np
import pytest

from src import audio
from src import config as C


@pytest.mark.parametrize("snr", C.SNR_LEVELS_DB)
def test_noise_is_scaled_to_requested_snr(snr):
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 200, C.CLIP_SAMPLES)).astype(np.float32)
    noisy = audio.add_white_noise(clean, snr, rng.standard_normal(C.CLIP_SAMPLES))
    measured = 10 * np.log10(audio.signal_power(clean) / audio.signal_power(noisy - clean))
    assert measured == pytest.approx(snr, abs=1e-3)


def test_background_noise_snr_uses_given_speech_power():
    rng = np.random.default_rng(1)
    clean, track = rng.standard_normal(C.CLIP_SAMPLES), rng.standard_normal(5 * C.CLIP_SAMPLES)
    noisy = audio.add_background_noise(clean, track, 10, offset=1234, clean_power=2.0)
    assert 10 * np.log10(2.0 / audio.signal_power(noisy - clean)) == pytest.approx(10, abs=1e-3)


def test_fix_length_pads_and_crops():
    assert len(audio.fix_length(np.ones(9000))) == C.CLIP_SAMPLES
    assert len(audio.fix_length(np.ones(20000))) == C.CLIP_SAMPLES


def test_log_mel_shape_single_and_batch():
    y = np.zeros((3, C.CLIP_SAMPLES), dtype=np.float32)
    assert audio.log_mel(y[0]).shape == (C.N_MELS, C.N_FRAMES)
    assert audio.log_mel(y).shape == (3, C.N_MELS, C.N_FRAMES)
