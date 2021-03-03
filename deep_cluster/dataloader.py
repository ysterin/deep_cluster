import torch
from torch import utils
from torch.utils import data
import pandas as pd
import numpy as np
from pathlib import Path 

pd.set_option('mode.chained_assignment', None)

import numpy as np
import scipy.signal as sig
import pandas as pd


def interpolate_indexes(data, idxs):
    segments = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
    for seg in segments:
        if len(seg) == 0:
            continue
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


def clear_low_likelihood(data, likelihood, thresh=0.8):
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

def read_df(df_file):
    df = pd.read_hdf(df_file)
    df.columns = df.columns.droplevel(0)
    df.index.name = 'index'
    df.index = df.index.map(int)
    df = df.applymap(float)
    return df


'''
process dataframe by smoothing the x and y coordinates
'''
def process_df(df):
    body_parts = pd.unique([col[0] for col in df.columns])
    smoothed_data = {}
    for part in body_parts:
        smoothed_data[(part, 'x')] = preproc(df[part].x, df[part].likelihood)
        smoothed_data[(part, 'y')] = preproc(df[part].y, df[part].likelihood)
        smoothed_data[(part, 'likelihood')] = df[part].likelihood.copy()

    smooth_df = pd.DataFrame.from_dict(smoothed_data)
    return smooth_df


# extract and possibly normalize the coordinates to shared coordinate base - the tail-base at (0, 0), and nose on the Y axis. 
def extract_coordinates(df: pd.DataFrame, normalize: bool = True):
    N = len(df)
    xy_df = df.drop(axis=1, columns='likelihood', level=1)
    coords = xy_df.values.reshape(N, -1, 2)
    if not normalize:
        return coords
    base_tail_coords = xy_df.tailbase.values[:, np.newaxis, :]
    centered_coords = coords - base_tail_coords
    centered_nose_coords = xy_df.nose.values - xy_df.tailbase.values
    thetas = np.arctan2(centered_nose_coords[:, 1], centered_nose_coords[:, 0])
    rotation_matrices = np.stack([np.stack([np.cos(thetas), np.sin(thetas)], axis=-1),
                                  np.stack([np.sin(thetas), -np.cos(thetas)], axis=-1)], axis=-1)
    normalized_coords = np.einsum("nij,nkj->nki", rotation_matrices, centered_coords)
    return normalized_coords


# A dataset of landmarks
# args: landmarks file: .h5 file of landmarks, from DeepLabCut
class LandmarkDataset(data.Dataset):
    def __init__(self, landmarks_file, normalize=True):
        super(LandmarkDataset, self).__init__()
        self.file = landmarks_file
        self.df = read_df(landmarks_file)
        self.df = process_df(self.df)
        self.coords = extract_coordinates(self.df, normalize)
        self.body_parts = pd.unique([col[0] for col in self.df.columns])
        
    def __getitem__(self, idx):
        return self.coords[idx]
    
    def __len__(self):
        return self.coords.shape[0]


class FeaturesDataset(data.Dataset):
    def __init__(self, landmarks_file, normalize=True):
        super(FeaturesDataset, self).__init__()
        self.file = landmarks_file
        self.df = read_df(landmarks_file)
        self.df = process_df(self.df)
        self.features = extract_features(self.df)
        # self.coords = extract_coordinates(self.df, normalize)
        # self.body_parts = pd.unique([col[0] for col in self.df.columns])

    def __getitem__(self, idx):
        return self.features[idx]

    def __len__(self):
        return self.features.shape[0]

    @property
    def n_features(self):
        return self.features.shape[1]


def calc_wavelet_transform(feature_data, min_width=12, max_width=120, n_waves=25):
    wavelet_widths = np.logspace(np.log10(min_width), np.log10(max_width), n_waves)
    transformed = sig.cwt(feature_data, sig.morlet2, widths=wavelet_widths)
    transformed /= np.sqrt(wavelet_widths[:,np.newaxis])
    return np.abs(transformed)


# A dataset of wavelets of landmarks.
# args: landmarks file: .h5 file of landmarks, from DeepLabCut
class LandmarkWaveletDataset(utils.data.Dataset):
    def __init__(self, landmarks_file, normalize=True, data=None, eps=3e-2):
        super(LandmarkWaveletDataset, self).__init__()
        self.file = landmarks_file
        self.normalize = normalize
        self.landmarks = LandmarkDataset(self.file, normalize)
        if data is None:
            coords = sig.decimate(self.landmarks.coords, q=4, axis=0)
            coords = coords.reshape((len(coords), -1))
            self.data = [calc_wavelet_transform(feat_data, min_width=2, max_width=60, n_waves=20) for feat_data in coords.T]
            self.data = np.concatenate(self.data, axis=0).T
            self.data /= (self.data.sum(axis=0, keepdims=True) + 1e-8)
#             self.data = np.sqrt(self.data)
            self.data = np.log(self.data + 1e-6).astype(np.float32) - 12
        else:
            self.data = data
        
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return LandmarkWaveletDataset(self.file, self.normalize, data=self.data[idx])
        return self.data[idx]
    
    def __len__(self):
        return self.data.shape[0]

    
    
'''
A class for sequence data. Each item of the dataset id a sequence of length 'seqlen' from the 'data' time-series.
for example, first item is data[0: seqlen], second is data[step: step + seqlen] etc.
Args:
    data: sequential data, for example numpy array, where the first dimension is time.
    seqlen: length of each sequence item.
    step: jump between one item and the next.
    diff: if True, each item is made of differences between consecutive time steps. Therefore, length of each item is actually seqlen - 1
          if False, each item is taken as is from the timeseries.
    flatten: whether to flatten each item to 1D or not. 
        If True, each item is 1D, which can be used in normal (non-recurrent or convolutional) Autoencoder.
        If False, each item is 2D [timesteps, data_dimension], for use in convolutional or recurrent Autoencoder.
'''
class SequenceDataset(data.Dataset):
    eps = 1e-8
    def __init__(self, data, seqlen=60, step=10, diff=False, flatten=True):
        super(SequenceDataset, self).__init__()
        self.seqlen, self.step, self.diff, self.flatten = seqlen, step, diff, flatten
        self.data = data
        self.mean, self.std = self._mean(), self._std()

    # calculates mean for standardization
    def _mean(self):
        if self.diff:
            mean = np.zeros_like(self.data[0])
        else:
            mean = self.data.mean(axis=0)
        return mean 

    # calculates standard deviation for standardization
    def _std(self):
        if self.diff:
            std = np.diff(self.data, axis=0).std(axis=0)
        else:
            std = self.data.std(axis=0)
        return std


    def __len__(self):
        return (len(self.data) - self.seqlen) // self.step

    def __getitem__(self, i):
        offset = self.step * i
        item = self.data[offset: offset + self.seqlen]
        if self.diff:
            item = np.diff(item, axis=0)
        item = (item - self.mean) / (self.std + self.eps)
        if self.flatten:
            item = np.reshape(item, (-1, ))
        return item

    def get_slice(self, start, end, step=1):
        item = self.data[start : end : step]
        if self.diff:
            item = np.diff(item, (-1, ))
        item = (item - self.mean) / (self.std + self.eps)
        return item



def main():
    data_path = Path('/home/orel/Data/K6/2020-03-30/Down/')
    landmarks_file = data_path / '0014DeepCut_resnet50_DownMay7shuffle1_1030000.h5'
    dataset = LandmarkDataset(landmarks_file)


if __name__ == "__main__":
    main()
