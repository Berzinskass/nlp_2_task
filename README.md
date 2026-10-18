# Task 2: 10-word speech recognition under noise

Everything is in one script, `main.py`:

1. **Dataset**: 500 recordings of each of 10 words (*yes, no, up, down, left, right,
   on, off, stop, go*) from Google Speech Commands v0.02, split by speaker:
   20 % of the speakers are used for testing.
2. **Features**: log-mel spectrogram (25 ms Hann window, 10 ms hop, 512-point FFT,
   40 mel filters), 40 x 101 per 1-second clip.
3. **Model**: small CNN (2 conv layers + 2 fully connected layers), trained on
   clean speech only.
4. **Testing**: clean speech, white Gaussian noise and real background noise
   (from the dataset's `_background_noise_` folder) at 30, 20, 10 and 0 dB SNR.

## Run

```bash
pip install -r requirements.txt
python main.py | tee results/output.txt
```

The first run downloads the dataset (2.4 GB) into `data/`. The script takes about
5 minutes on a laptop CPU. Printed results are saved in `results/output.txt` and
figures in `results/`.
