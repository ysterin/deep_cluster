import numpy as np
import torch
from torch import utils
import pandas as pd
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from scipy import signal as sig
import os
from pathlib import Path
import re
from torch.utils import data
import pandas as pd
import numpy as np
from pathlib import Path
from dataloader import LandmarkDataset, SequenceDataset, LandmarkWaveletDataset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, confusion_matrix, accuracy_score

pd.set_option('mode.chained_assignment', None)


# A simple autoencoder
class Autoencoder(nn.Module):
    # n_neurons: the sizes of each layer in the encoder - the decoder has the same number of neurons in each layer, in reverse.
    def __init__(self, n_neurons, batch_norm=False, loss_func=None, dropout=0.):
        super(Autoencoder, self).__init__()
        if loss_func:
            self.loss_func = loss_func
        else:
            self.loss_func = F.mse_loss
        n_layers = len(n_neurons) - 1
        layers = list()
        for i in range(n_layers):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
            if i+1 < n_layers:
                layers.append(nn.ELU())
                if batch_norm:
                    layers.append(nn.BatchNorm1D(n_neurons[i+1]))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*layers)
        layers = list()
        n_neurons = n_neurons[::-1]
        for i in range(n_layers):
            layers.append(nn.Linear(n_neurons[i], n_neurons[i+1]))
            if i+1 < n_layers:
                layers.append(nn.ELU())
                if batch_norm:
                    layers.append(nn.BatchNorm1D(n_neurons[i+1]))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.decoder(self.encoder(x))
    
    # computes auto encoding loss for training and evaluation
    def shared_step(self, bx):
        z = self.encoder(bx)
        out = self.decoder(z)
        loss = self.loss_func(bx, out)
        return loss
    
    # encode data to the hidden dimension.
    def encode(self, data, batch_size=256):
        dl = DataLoader(data, batch_size=batch_size, shuffle=False)
        X = []
        self.cuda()
        with torch.no_grad():
            for bx in dl:
                x_encoded = self.encoder(bx.cuda())
                X.append(x_encoded.cpu().numpy())
        X_encoded = np.concatenate(X)
        return X_encoded
        
    # cluster the data using K Means on the encoded data. The data is in the form of SequenceDataset.
    def cluster(self, data, batch_size=256, n_clusters=30):
        X_encoded = self.encode(data, batch_size=batch_size)
        kmeans = KMeans(n_clusters=n_clusters)
        labels = kmeans.fit_predict(X_encoded)
        return labels
    
    
#pytorch-lightning module for training the autoencoder
class PLAutoencoder(pl.LightningModule):
    def __init__(self, landmark_files, n_neurons=[203, 128, 128, 7], lr=1e-3, seqlen=30, patience=20, batch_norm=False, wd=0.05):
        super(PLAutoencoder, self).__init__()
        self.landmark_files = landmark_files
        self.seqlen = seqlen
        self.hparams = {'lr': lr, 'patience': patience, 'wd': wd}
        self.model = Autoencoder(n_neurons, batch_norm)

    def forward(self, x):
        return self.model(x)
    
    def prepare_data(self):
        landmark_datasets = []
        for file in self.landmark_files:
            try:
                ds = LandmarkDataset(file)
                landmark_datasets.append(ds)
            except OSError:
                pass
        coords = [sig.decimate(ds.coords, q=4, axis=0).astype(np.float32) for ds in landmark_datasets]
        N, n_coords, _ = coords[0].shape
        self.coords = coords
        train_data = [crds[:int(0.8*crds.shape[0])].reshape(-1, n_coords*2) for crds in coords]
        valid_data = [crds[int(0.8*crds.shape[0]):].reshape(-1, n_coords*2) for crds in coords]
        train_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=1, diff=False) for data in train_data]
        valid_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=10, diff=False) for data in valid_data]
        self.train_ds = ConcatDataset(train_dsets)
        self.valid_ds = ConcatDataset(valid_dsets)
        all_data = [crds.reshape(-1, n_coords*2) for crds in coords]
        all_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=1, diff=False) for data in all_data]
        self.all_ds = ConcatDataset(all_dsets)
        

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=4)

    def val_dataloader(self):
        # dataset = SequenceDataset(X_val, seqlen=30, step=5, diff=True)
        return DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=4)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=self.hparams['wd'])
#         sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2 ,patience=self.hparams['patience'], verbose=True, min_lr=1e-6)
#         scheduler = {'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_checkpoint_on', 'reduce_on_plateau': True}
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.85**(epoch//10))
        return [opt], [sched]    
#     return [opt], [scheduler]
    
    def training_step(self, batch, batch_idx):
        loss = self.model.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss
    
    
def kl_div(p, q):
    return (p * (p.log() - q.log())).sum(dim=-1)
    
    
def DKL_softmax(bx, bx_recon):
    bx_probs = F.softmax(bx, dim=-1)
    recon_probs = F.softmax(bx_recon, dim=-1)
    return kl_div(bx_probs, recon_probs).mean()
   
    
#pytorch-lightning module for training the autoencoder
class PLWaveletAutoencoder(pl.LightningModule):
    def __init__(self, landmark_files, n_neurons=[480, 128, 128, 7], lr=1e-3, n_wavelets=20, patience=20, batch_norm=False, wd=0.05):
        super(PLWaveletAutoencoder, self).__init__()
        self.landmark_files = landmark_files
        self.n_wavelets = n_wavelets
        self.hparams = {'lr': lr, 'patience': patience, 'wd': wd}
        self.model = Autoencoder(n_neurons, batch_norm, loss_func=DKL_softmax)

    def forward(self, x):
        return self.model(x)
    
    def prepare_data(self):
        datasets = []
        for file in self.landmark_files:
            try:
                ds = LandmarkWaveletDataset(file)
                datasets.append(ds)
            except OSError:
                pass
        train_dsets = [ds[:int(0.8*len(ds))] for ds in datasets]
        valid_dsets = [ds[int(0.8*len(ds)):] for ds in datasets]
        self.train_ds = ConcatDataset(train_dsets)
        self.valid_ds = ConcatDataset(valid_dsets)
        self.all_ds = ConcatDataset(datasets)  
        self.datasets = datasets

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=self.hparams['wd'])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.85**(epoch//10))
        return [opt], [sched]    
    
    def training_step(self, batch, batch_idx):
        loss = self.model.shared_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.shared_step(batch)
        self.log('val_loss', loss, prog_bar=True)
        return loss