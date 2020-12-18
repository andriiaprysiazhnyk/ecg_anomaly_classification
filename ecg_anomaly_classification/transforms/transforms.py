import torch
import numpy as np
from scipy import signal


def spectrogram(ecg, window_size, n_overlap):
    fs = 100
    res = None

    for i in range(ecg.shape[1]):
        _, _, sxx = signal.spectrogram(ecg[:, i], fs, nperseg=window_size, noverlap=n_overlap)
        res = sxx.T if res is None else np.concatenate((res, sxx.T), axis=1)

    return res


def scalogram(ecg, max_width, add_time_domain):
    res = None

    for i in range(ecg.shape[1]):
        sxx = signal.cwt(ecg[:, i], signal.ricker, np.arange(1, max_width + 1))
        res = sxx.T if res is None else np.concatenate((res, sxx.T), axis=1)

    if add_time_domain:
        res = np.concatenate((res, ecg), axis=1)

    return res


def get_transform(config):
    if config["type"] == "spectrogram":
        window_size = config.get("window_size", 100)
        n_overlap = config.get("n_overlap", window_size // 8)
        return lambda x: torch.FloatTensor(spectrogram(x, window_size, n_overlap))
    elif config["type"] == "scalogram":
        return lambda x: torch.FloatTensor(scalogram(x, config["max_width"], config["add_time_domain"]))

    raise TypeError(f"Unknown transform type: {config['type']}")


def get_output_dim(config):
    if config["type"] == "spectrogram":
        window_size = config.get("window_size", 100)
        return 12 * ((window_size if window_size % 2 == 1 else window_size + 1) // 2 + 1)
    elif config["type"] == "scalogram":
        return 12 * config["max_width"] + (12 if config["add_time_domain"] else 0)

    raise TypeError(f"Unknown transform type: {config['type']}")
