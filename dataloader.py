import torch
from torch.utils import data
import pandas as pd
import numpy as np
from smooth import preproc
from pathlib import Path 

pd.set_option('mode.chained_assignment', None)


def read_df(df_file):
    df = pd.read_hdf(df_file)
    df.columns = df.columns.droplevel(0)
    df.index.name = 'index'
    df.index = df.index.map(int)
    df = df.applymap(float)
    return df


def process_df(df):
    body_parts = pd.unique([col[0] for col in df.columns])
    smoothed_data = {}
    for part in body_parts:
        smoothed_data[(part, 'x')] = preproc(df[part].x, df[part].likelihood)
        smoothed_data[(part, 'y')] = preproc(df[part].y, df[part].likelihood)
        smoothed_data[(part, 'likelihood')] = df[part].likelihood.copy()

    smooth_df = pd.DataFrame.from_dict(smoothed_data)
    return smooth_df


def normalize_coordinates(df: pd.DataFrame):
    N = len(df)
    xy_df = df.drop(axis=1, columns='likelihood', level=1)
    coords = xy_df.values.reshape(N, -1, 2)
    base_tail_coords = xy_df.tailbase.values[:, np.newaxis, :]
    centered_coords = coords - base_tail_coords
    centered_nose_coords = xy_df.nose.values - xy_df.tailbase.values
    thetas = np.arctan2(centered_nose_coords[:, 1], centered_nose_coords[:, 0])
    rotation_matrices = np.stack([np.stack([np.cos(thetas), np.sin(thetas)], axis=-1),
                                  np.stack([np.sin(thetas), -np.cos(thetas)], axis=-1)], axis=-1)
    normalized_coords = np.einsum("nij,nkj->nki", rotation_matrices, centered_coords)
    return normalized_coords


class LandmarkDataset(data.Dataset):
    def __init__(self, landmarks_file):
        super(LandmarkDataset, self).__init__()
        self.file = landmarks_file
        self.df = read_df(landmarks_file)
        self.df = process_df(self.df)
        self.coords = normalize_coordinates(self.df)
        self.body_parts = pd.unique([col[0] for col in self.df.columns])
        
    def __getitem__(self, idx):
        return self.coords[idx]
    
    def __len__(self):
        return self.coords.shape[0]

    
class SequenceDataset(data.Dataset):
    eps = 1e-8
    def __init__(self, data, seqlen=60, step=10, diff=True, flatten=True):
        super(SequenceDataset, self).__init__()
        self.seqlen, self.step, self.diff, self.flatten = seqlen, step, diff, flatten
        self.data = data
        self.mean, self.std = self._mean(), self._std()

    def _mean(self):
        if self.diff:
            mean = np.zeros_like(self.data[0])
        else:
            mean = self.data.mean(axis=0)
        return mean 

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



def main():
    data_path = Path('/home/orel/Data/K6/2020-03-30/Down/')
    landmarks_file = data_path / '0014DeepCut_resnet50_DownMay7shuffle1_1030000.h5'
    dataset = LandmarkDataset(landmarks_file)


if __name__ == "__main__":
    main()