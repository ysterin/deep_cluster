import torch
from torch import utils
from torch.utils import data
import pandas as pd
import numpy as np
from pathlib import Path
import bisect
import cv2 as cv
import re
from torch.utils.data.dataset import ConcatDataset, Subset, TensorDataset
from torch.utils.data.dataloader import DataLoader
pd.set_option('mode.chained_assignment', None)
from itertools import chain
import numpy as np
import scipy.signal as sig
import pandas as pd
import pytorch_lightning as pl


'''
Interpolate between the values in idxs.
data: timeseries data
idxs: problematic idxs to interpolate over, for example if likelihood is small or there are abrupt jumps.
'''
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


# Interpolate over masked values.
def interpolate_mask(data, mask):
    idxs = np.where(mask)[0]
    return interpolate_indexes(data, idxs)


def clear_low_likelihood(data, likelihood, thresh=0.8):
    return interpolate_mask(data, likelihood < thresh)


# find outlier values with high pass filter and interploate over them using 
def clear_outliers(data, hpf_freq=30, fs=240):
    cutoff = fs / hpf_freq
    filt = sig.butter(4, hpf_freq, btype='high', output='ba', fs=fs)
    filtered = sig.filtfilt(*filt, data)
    return interpolate_mask(data, np.abs(filtered) > cutoff)


# smooth data with low pass filter
def smooth(data, lpf_freq=20, fs=240):
    filt = sig.butter(4, lpf_freq, btype='low', output='ba', fs=fs)
    filtered = sig.filtfilt(*filt, data)
    return filtered


# preprocess (landmarks) timeseries - interpolate over low likelihood segments and outliers and abrupt jumps, and smooth with low-pass filter
def preproc(data, likelihood=None, fps=120):
    if not likelihood is None:
        data = clear_low_likelihood(data, likelihood)
    return smooth(clear_outliers(data, fs=fps), fs=fps)

import os


# find the video file for the landmarks file, assuming it is in the same folder.
def find_video_file(df_file: Path):
    try:
        video_file = list(df_file.parent.glob(r'00*.MP4'))[0]
    except IndexError:
        raise Exception(f"video file does no exist in {df_file.parent}")
#     video_file = file_dir / f'{file_id}.MP4'
    if len(list(df_file.parent.glob(r'00*.MP4'))) > 1:
        raise Exception('more than one video file in folder')
    assert os.path.exists(video_file), f'file {video_file} does not exist!'
    return video_file


# load landmarks file 
def read_df(df_file):
    df = pd.read_hdf(df_file)
    if len(df.columns.levels) > 2:
        df.columns = df.columns.droplevel(0)
    df.index.name = 'index'
    df.index = df.index.map(int)
    df = df.applymap(float)
    df.attrs['file'] = df_file
    df.attrs['video_file'] = find_video_file(df_file)
    df.attrs['fps'] = 120
    return df


'''
process dataframe by smoothing the x and y coordinates
'''
def process_df(df, fps=120):
    body_parts = df.columns.levels[0]
    # body_parts = pd.unique([col[0] for col in df.columns])
    smoothed_data = {}
    if 'likelihood' in df.columns.levels[1]:
        for part in body_parts:
            smoothed_data[(part, 'x')] = preproc(df[part].x, df[part].likelihood, fps=fps)
            smoothed_data[(part, 'y')] = preproc(df[part].y, df[part].likelihood, fps=fps)
            smoothed_data[(part, 'likelihood')] = df[part].likelihood.copy()
    else:
        for part in body_parts:
            smoothed_data[(part, 'x')] = preproc(df[part].x, None, fps=fps)
            smoothed_data[(part, 'y')] = preproc(df[part].y, None, fps=fps)

    smooth_df = pd.DataFrame.from_dict(smoothed_data)
    smooth_df.attrs = df.attrs
    smooth_df.columns = df.columns
    return smooth_df

'''
rotate the coordinates in the data frame to standard coordinate system, with the tailbase at (0,0).
args:
    df: data frame with coordinates.
'''
def standardize_df(df, lock_theta=False):
    df = df.copy()
    N = len(df)
    body_parts = df.columns.levels[0]
    # body_parts = pd.unique([col[0] for col in df.columns])
    if 'likelihood' in df.columns.remove_unused_levels().levels[1]:
        xy_df = df.drop(axis=1, columns='likelihood', level=1)
    else:
        xy_df = df
    coords = xy_df.values.reshape(N, -1, 2)
    base_tail_coords = xy_df.tailbase.values[:, np.newaxis, :]
    centered_coords = coords - base_tail_coords
    centered_nose_coords = xy_df.nose.values - xy_df.tailbase.values
    thetas = np.arctan2(centered_nose_coords[:, 1], centered_nose_coords[:, 0])
    if lock_theta:
        thetas = thetas.mean(axis=0).repeat(N, axis=0)
    rotation_matrices = np.stack([np.stack([np.cos(thetas), np.sin(thetas)], axis=-1),
                                  np.stack([np.sin(thetas), -np.cos(thetas)], axis=-1)], axis=-1)
    normalized_coords = np.einsum("nij,nkj->nki", rotation_matrices, centered_coords)
    for i, part in enumerate(body_parts):
        df[part]['x'] = normalized_coords[:,i,0]
        df[part]['y'] = normalized_coords[:,i,1]
    return df

'''
normalize the coordinates in dataframe to have mean of 0 and std of 1.
'''
def normalize_df(df, means=None, stds=None):
    body_parts = df.columns.levels[0]
    # body_parts = pd.unique([col[0] for col in df.columns])
    if means is None:
        means = df.mean()
        stds = df.std()
    for part in body_parts:
        df[(part, 'x')] = (df[(part, 'x')] - means[part]['x']) / (stds[part]['x'] + 1e-8)
        df[(part, 'y')] = (df[(part, 'y')] - means[part]['y']) / (stds[part]['y'] + 1e-8)
    return df

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

# find cotiguous segments with high likelihood in the landmarks timeseries. 
def find_segments(df, threshold=0.95, min_length=60):
    df = df.copy()
    body_parts = df.columns.levels[0]
    confidence = np.prod(np.stack([df[bp].likelihood.values for bp in body_parts]), axis=0)
    idxs = np.where(confidence > threshold)[0]
    segments = np.split(idxs, np.where(np.diff(idxs) > 1)[0] + 1)
    segments = [seg for seg in segments if len(seg) > min_length]
    return segments

# A dataset of landmarks
# args: landmarks file: .h5 file of landmarks, from DeepLabCut
class LandmarkDataset(data.Dataset):
    def __init__(self, landmarks_file, normalize=True, fps=None, smooth=True):
        super(LandmarkDataset, self).__init__()
        self.file = Path(landmarks_file)
        if not fps:
            if self.file.parent.parent.parent.name == 'K7':
                fps = 120
            elif self.file.parent.parent.parent.name == 'K6':
                fps = 240
            else:
                raise Exception(f"self.file.parent.parent.parent.name is {self.file.parent.parent.parent.name}")
        self.df = read_df(landmarks_file)
        self.fps = fps
        if smooth:
            self.df = process_df(self.df, fps=fps)
        self.coords = extract_coordinates(self.df, normalize)
        self.body_parts = self.df.columns.levels[0]
        
    def __getitem__(self, idx):
        return self.coords[idx]
    
    def __len__(self):
        return self.coords.shape[0]

'''
A dataset of fratures for landmarks.
Needs "extract_features" function to extract the features.
'''
# class FeaturesDataset(data.Dataset):
#     def __init__(self, landmarks_file, normalize=True):
#         super(FeaturesDataset, self).__init__()
#         self.file = landmarks_file
#         self.df = read_df(landmarks_file)
#         self.df = process_df(self.df)
#         self.features = extract_features(self.df)
#         # self.coords = extract_coordinates(self.df, normalize)
#         # body_parts = df.columns.levels[0]

#     def __getitem__(self, idx):
#         return self.features[idx]

#     def __len__(self):
#         return self.features.shape[0]

#     @property
#     def n_features(self):
#         return self.features.shape[1]


def calc_wavelet_transform(feature_data, min_width=12, max_width=120, n_waves=25):
    wavelet_widths = np.logspace(np.log10(min_width), np.log10(max_width), n_waves)
    transformed = sig.cwt(feature_data, sig.morlet2, widths=wavelet_widths)
    transformed /= np.sqrt(wavelet_widths[:,np.newaxis])
    return np.abs(transformed)


# A dataset of wavelets of landmarks.
# args: landmarks file: .h5 file of landmarks, from DeepLabCut
'''
A dataset of wavelets of landmarks.
args: landmarks file: .h5 file of landmarks, from DeepLabCut or HourGlass
      normalize: wether to normalize the landmraks first (to a single coordinate frame)
      data: used for slicing the dataset, should always be None (the default) when calling __init__
'''
class LandmarkWaveletDataset(utils.data.Dataset):
    def __init__(self, landmarks_file, normalize=True, data=None):
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
    def __init__(self, data_frame, seqlen=60, step=10, diff=False, flatten=True):
        super(SequenceDataset, self).__init__()
        self.seqlen, self.step, self.diff, self.flatten = seqlen, step, diff, flatten
        self.data_frame = data_frame
        self.data = data_frame.values.astype(np.float32)
        self.mean, self.std = 0., 1.
        # self.mean, self.std = self._mean(), self._std()

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

    def get_indexes(self, i):
        offset = self.step * i
        return self.data_frame.index[offset: offset + self.seqlen]

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


# Find the original dataset for idx, from a concatenation of datasts (ConcatDataset)
def find_sub_dataset_idx(ds: ConcatDataset, idx):
    assert isinstance(ds, ConcatDataset)
    dataset_idx = bisect.bisect_right(ds.cumulative_sizes, idx)
    if dataset_idx == 0:
        sample_idx = idx
    else:
        sample_idx = idx - ds.cumulative_sizes[dataset_idx - 1]
    return dataset_idx, sample_idx


# find the video file and frame number for a specific timestep (idx) in the landmarks dataset (ds) 
def find_frame(ds, idx):
    ds_id, idx = find_sub_dataset_idx(ds, idx)
    ds = ds.datasets[ds_id]
    if isinstance(ds, Subset):
        ds, idx = ds.dataset, ds.indices[idx]
    if isinstance(ds, ConcatDataset):
        ds_id, idx = find_sub_dataset_idx(ds, idx)
        ds = ds.datasets[ds_id]
    video_file = ds.data_frame.attrs['video_file']
    frame_idxs = ds.get_indexes(idx)
    return video_file, frame_idxs

# filter segments by energy
def filter_segments_by_energy(min_energy=1e-7):
    def filter(segment_df):
        data = segment_df.values.astype(np.float32)
        ff, Pxx = sig.periodogram(data.T, fs=segment_df.attrs['fps'])
        energy = Pxx.mean()
        return energy > min_energy
    return filter

# drop unused parts from the dataframe.
def drop_parts(df, parts):
    if not parts:
        return df
    parts = [part for part in parts if part in df.columns.levels[0]]
    df = df.drop(labels=parts, axis=1, level=0)
    df.columns = df.columns.remove_unused_levels()
    return df


'''
Pytorch Lightning DataModule for landmarks data, contains both training and validation datasets.
'''
class LandmarksDataModule(pl.LightningDataModule):
    def __init__(self, landmark_files, fps=120, seqlen=60, step=30, bs=32, to_drop=['tail2'], filter_by_likelihood=False):
        super(LandmarksDataModule, self).__init__()
        self.landmark_files = landmark_files
        self.bs = bs
        self.fps, self.seqlen, self.step = fps, seqlen, step
        self.to_drop = to_drop
        self.filter_by_likelihood = filter_by_likelihood

    '''
    read the landmarkk files and preprocess them.
    can filter the data according to likelihood.
    '''
    def prepare_data(self, *args, **kwargs):
        data_frames = list(map(read_df, self.landmark_files))
        self.raw_data = data_frames
        data_frames = map(lambda df: drop_parts(df, self.to_drop), data_frames)
        data_frames = map(lambda df: process_df(df, self.fps), data_frames)
        data_frames = list(map(standardize_df, data_frames))
        all_df = pd.concat(data_frames)
        data_frames = list(map(lambda df: normalize_df(df, all_df.mean(), all_df.std()), data_frames))
        if self.filter_by_likelihood and 'likelihood' in data_frames[0].columns.levels[1]:
            segments_list = list(map(find_segments, data_frames))
            self.segments_list = segments_list
            data_frames = list(map(lambda df:  df.drop(axis=1, columns='likelihood', level=1), data_frames))
            segment_dfs = [[df.iloc[seg] for seg in segments] for df, segments in zip(data_frames, segments_list)]
            sequence_datasets = [[SequenceDataset(df, seqlen=self.seqlen, step=self.step, diff=False) for df in dfs]
                                     for dfs in segment_dfs]
            datasets = [ConcatDataset(dsets) for dsets in sequence_datasets]
        else:
            if 'likelihood' in data_frames[0].columns.levels[1]:
                data_frames = list(map(lambda df: df.drop(columns='likelihood', axis=1, level=1), data_frames))
            datasets = [SequenceDataset(df, seqlen=self.seqlen, step=self.step, diff=False) for df in data_frames]
        train_dsets = [Subset(ds, range(0, int(0.8 * len(ds)))) for ds in datasets]
        valid_dsets = [Subset(ds, range(int(0.8 * len(ds)), len(ds))) for ds in datasets]
        self.train_ds = ConcatDataset(train_dsets)
        self.valid_ds = ConcatDataset(valid_dsets)
        self.all_ds = ConcatDataset(datasets)
        self.data_frames = data_frames
        self.datasets = datasets
        self.all_df = all_df
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.bs, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(self.valid_ds, batch_size=self.bs, shuffle=False)


def main():
    data_path = Path('/home/orel/Data/K6/2020-03-30/Down/')
    landmarks_file = data_path / '0014DeepCut_resnet50_DownMay7shuffle1_1030000.h5'
    dataset = LandmarkDataset(landmarks_file)


if __name__ == "__main__":
    main()
