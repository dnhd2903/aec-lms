"""
sim/audio.py — WAV read/write via scipy.io.wavfile (no soundfile dependency).
"""

import warnings
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def load_wav(path, fs: int) -> np.ndarray:
    """
    Read a WAV file and return a mono float64 signal in [−1, 1] at sample rate fs.

    - Integer formats (int16/int32/uint8) are scaled to [−1, 1].
    - Stereo is mixed down to mono by averaging the channels.
    - If the file's sample rate differs from fs, it is resampled (resample_poly
      includes the anti-aliasing filter).
    """
    fs_in, data = wavfile.read(path)

    if data.dtype.kind == "i":
        data = data / float(np.iinfo(data.dtype).max)
    elif data.dtype.kind == "u":
        data = (data.astype(np.float64) - 128.0) / 128.0
    data = data.astype(np.float64)

    if data.ndim == 2:
        data = data.mean(axis=1)

    if fs_in != fs:
        g = gcd(fs_in, fs)
        data = resample_poly(data, fs // g, fs_in // g)

    return data


def save_wav(path, x: np.ndarray, fs: int) -> None:
    """
    Write a float signal in [−1, 1] as a 16-bit WAV.

    No automatic normalisation: when comparing mic.wav with aec_output.wav, the
    loudness difference is exactly what you need to hear. Samples beyond ±1 are
    clipped with a warning.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(x)) if len(x) else 0.0
    if peak > 1.0:
        warnings.warn(f"{path.name}: peak amplitude {peak:.2f} > 1, samples beyond ±1 will be clipped")
    wavfile.write(path, fs, (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16))
