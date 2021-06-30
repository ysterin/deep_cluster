import numpy as np
import torch
from torch import utils
import pandas as pd
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn 
from torch.nn import functional as F
import pytorch_lightning as pl
from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal as sig
import os
from pathlib import Path
import re
from torch.utils import data
import random
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from deep_cluster.dataloader import LandmarkDataset, SequenceDataset, LandmarkWaveletDataset
from deep_cluster.simple_autoencoder import Autoencoder, PLAutoencoder
from deep_cluster.dataloader import LandmarksDataModule

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, accuracy_score


random.seed(32)
np.random.seed(32)
torch.random.manual_seed(32)


data_root = Path('/mnt/Storage1/Data/K7')

landmark_files = list(data_root.glob('2020-*/Down/model=*.h5'))

seqlen = 60
dm = LandmarksDataModule(landmark_files, seqlen=seqlen, step=3, to_drop=[])
dm.prepare_data()
n_parts = len(dm.data_frames[0].columns.levels[0])
model = PLAutoencoder(landmark_files, n_neurons=[2*n_parts*seqlen, 1024, 512, 128, 32, 4], lr=3e-4, patience=20, dropout=0.0)
trainer = pl.Trainer(gpus=1, progress_bar_refresh_rate=10, max_epochs=2, logger=pl.loggers.WandbLogger("landmarks autoencoder"), log_every_n_steps=1)
trainer.fit(model, dm)

df_file = Path('triplets/data/robust_triplets.csv')

df = pd.read_csv(df_file).dropna()
df['file_id'] = df.video_file.map(lambda p: int(re.search(r'00\d\d', os.path.split(p)[-1])[0]))

import cv2 as cv
from landmarks_video import LandmarksVideo
from triplets_gui import Animation, ClipsDisplay, VerificationApp
import landmarks_video
import triplets_gui

from deep_cluster.simple_autoencoder import Autoencoder, PLAutoencoder
from deep_cluster import dataloader
seqlen = 60
data_root = Path('/mnt/Storage1/Data/K7')

landmark_files = list(data_root.glob('2020-*/Down/model=*0044.h5'))
dm = dataloader.LandmarksDataModule(landmark_files, step=1, seqlen=seqlen, to_drop=[])
dm.prepare_data()
n_parts = len(dm.data_frames[0].columns.levels[0])


X_encoded = model.model.encode(dm.all_ds)

def get_segment_enc(X_encoded, start, end):
    mid = (start + end) // 2
    return X_encoded[mid - seqlen//2]
#     return X_encoded[start: end - seqlen].mean(axis=0)

def decode_segment_string(seg_string):
    start, end = seg_string[1:-1].split(',')
    return int(start), int(end)

def get_enc_from_string(seg_string):
    start, end = decode_segment_string(seg_string)
    return get_segment_enc(X_encoded, start, end)

# def get_landmarks_from_string(seg_string):
#     start, end = decode_segment_string(seg_string)
#     return raw_landmarks_dataset.get_slice(start, end).reshape(-1)

for sample in ['anchor', 'sample1', 'sample2']:
    df[f'{sample}_enc'] = df[sample].map(lambda seg: get_enc_from_string(seg))


df['d1'] = (df['anchor_enc'] - df['sample1_enc']).map(np.linalg.norm)
df['d2'] = (df['anchor_enc'] - df['sample2_enc']).map(np.linalg.norm)
df['d3'] = (df['sample1_enc'] - df['sample2_enc']).map(np.linalg.norm)
df['pred_selected'] = (df['d1'] <= df['d2']).map(lambda b: 1 if b else 2)
df.selected = df.selected.map(int)
df['pred_correct'] = (df.selected == df.pred_selected)
df.drop(labels=['d1', 'd2', 'd3'], axis=1)
df.describe()
print(df[(df.selected != 0)]['pred_correct'].value_counts())