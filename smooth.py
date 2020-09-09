import numpy as np
import scipy.signal as sig
import pandas as pd


def interpolate_indexes(data, idxs):
    segments = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
    for seg in segments:
        if seg[0] == 0:
            data[seg[0]:seg[-1]+1] = data[seg[-1] + 1]
        elif seg[-1] == len(data) - 1:
            data[seg[0]:seg[-1]+1] = data[seg[0] - 1]
        else:
            data[seg[0]:seg[-1]+1] = np.linspace(data[seg[0] - 1], data[seg[-1] + 1], num=len(seg) + 1, endpoint=False)[1:]
    return data


def interpolate_mask(data, mask):
    idxs = np.where(mask)[0]
    return interpolate_indexes(data, idxs)


def clear_low_likelihood(data, likelihood, thresh=0.5):
    return interpolate_mask(data, likelihood < thresh)


def clear_outliers(data, hpf_freq=60, cutoff=4, fs=240):
    cutoff = fs / hpf_freq
    filt = sig.butter(4, hpf_freq, btype='high', output='ba', fs=fs)
    filtered = sig.filtfilt(*filt, data)
    return interpolate_mask(data, np.abs(filtered) > cutoff)


def smooth(data, lpf_freq=20, fs=240):
    filt = sig.butter(4, lpf_freq, btype='low', output='ba', fs=fs)
    filtered = sig.filtfilt(*filt, data)
    return filtered


def preproc(data, likelihood):
    data = clear_low_likelihood(data, likelihood)
    return smooth(clear_outliers(data))
