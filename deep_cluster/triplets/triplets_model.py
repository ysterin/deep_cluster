import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader, ConcatDataset, TensorDataset
from torch import distributions as D
from deep_cluster.dataloader import SequenceDataset, LandmarkDataset
from deep_cluster.VaDE_autoencoder import VaDE, SimpleAutoencoder
from scipy.optimize import linear_sum_assignment as linear_assignment
from torch.distributions import Normal, Laplace, kl_divergence, kl, Categorical
import pytorch_lightning as pl
from sklearn.mixture import GaussianMixture
from scipy import signal as sig
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
from torch.distributions import kl_divergence

landmarks_dir = Path("/mnt/storage2/shuki/data/THEMIS/")


def decode_seg_string(seg_string):
    start, end = seg_string[1:-1].split(',')
    start, end = int(start), int(end)
    start_idx, end_idx = start // 4 , end // 4 
    return (start_idx, end_idx)

def get_segment(landmarks_ds, segment_string, seqlen=30):
    start, end = decode_seg_string(segment_string)
    mid = (start + end) // 2
    start_idx, end_idx = mid - seqlen // 2, mid + seqlen // 2
    assert start_idx < end_idx, print(start_idx, end_idx)
    item = landmarks_ds.get_slice(start_idx, end_idx)
    # print (item.shape)
    # import pdb; pdb.set_trace()
    assert item.shape[0] == seqlen, print(start, end, start_idx, end_idx, len(landmarks_ds.data), item.shape)
    return np.reshape(item, (-1, ))

def ifnone(x, val):
    if x is None:
        return val
    return x


class TripletsDataset(Dataset):
    def __init__(self, triplets_file, landmark_datasets_dict, seqlen):
        super(TripletsDataset, self).__init__()
        self.triplets_file = triplets_file
        self.seqlen = seqlen
        self.df = pd.read_csv(triplets_file).dropna()
        self.df = self.df.loc[self.df.selected != '0']
        # print(self.df['anchor'].map(decode_seg_string).map(lambda o: o[1] - o[0] < 30).value_counts())
        self.df = self.df[self.df['anchor'].map(decode_seg_string).map(lambda o: o[1] < 100000)]
        # print(self.df['anchor'].map(decode_seg_string).map(lambda o: o[1] - o[0] < 30).value_counts())
        self.landmark_datasets_dict = {file.name[:4]: ds for file, ds in landmark_datasets_dict.items()}
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_file = row.video_file
        video_file_id = video_file.split('/')[-2]
        landmarks_ds = self.landmark_datasets_dict[video_file_id]
        # print([decode_seg_string(row[s]) for s in ['anchor', 'sample1', 'sample2']])
        segments_dict = {'anchor': row.anchor}
        if row.selected == '1':
            segments_dict['positive'] = row.sample1
            segments_dict['negative'] = row.sample2
        elif row.selected == '2':
            segments_dict['positive'] = row.sample2
            segments_dict['negative'] = row.sample1
        else:
            raise Exception('wrong selected option')
            pass
        return {sample: get_segment(landmarks_ds, segments_dict[sample], self.seqlen)\
                     for sample in ['anchor', 'positive', 'negative']}


class CombinedDataset(Dataset):
    def __init__(self, landmarks_dataset, triplets_dataset):
        super(CombinedDataset, self).__init__()
        self.landmarks_dataset = landmarks_dataset
        self.triplets_dataset = triplets_dataset

    def __len__(self):
        return len(self.landmarks_dataset)
    
    @property
    def n_triplets(self):
        return len(self.triplets_dataset)

    def __getitem__(self, idx):
        landmarks = self.landmarks_dataset[idx]
        triplet = self.triplets_dataset[idx % self.n_triplets]
        return (landmarks, triplet)


def normal_to_multivariate(p):
    return D.MultivariateNormal(p.mean, scale_tril=torch.diag_embed(p.stddev))

def cross_entropy(P, Q):
    try:
        return kl_divergence(P, Q) + P.entropy()
    except NotImplementedError:
        if type(P) == D.Independent and type(P.base_dist) == D.Normal:
            return kl_divergence(normal_to_multivariate(P), Q) + P.entropy()
        raise NotImplementedError

def kl_distance(P, Q):
    return 0.5 * (kl_divergence(P, Q) + kl_divergence(Q, P))



class TripletVaDE(pl.LightningModule):
    def __init__(self, n_neurons=[1440, 512, 256, 10],
                 batch_norm=False,
                 k=10, 
                 lr=1e-3, 
                 lr_gmm = None,
                 batch_size=256, 
                 device='cuda', 
                 pretrain_epochs=50, 
                 pretrained_model=None, 
                 triplet_loss_margin=0.5,
                 triplet_loss_alpha=0.,
                 triplet_loss_alpha_kl=0.,
                 warmup_epochs=10,
                 landmark_files=None,
                 triplets_file='deep_cluster/triplets/data/selected_triplets.csv', 
                 triplet_loss_margin_kl=1.,
                 triplets_random_state=None,
                 seqlen=30):
        super(TripletVaDE, self).__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.triplets_file = triplets_file
        self.seqlen = seqlen
        # self.hparams = {'lr': lr, 'lr_gmm': lr_gmm, 'triplet_loss_margin': triplet_loss_margin, 'triplet_loss_alpha': triplet_loss_alpha}
        # self.hparams['triplet_loss_margin_kl'] = triplet_loss_margin_kl
        # self.hparams['batch_size'] = batch_size
        # self.hparams['triplet_loss_alpha_kl'] = triplet_loss_alpha_kl
        # self.hparams['warmup_epochs'] = warmup_epochs
        self.landmark_files = landmark_files
        self.n_neurons, self.pretrain_epochs, self.batch_norm = n_neurons, pretrain_epochs, batch_norm
        init_gmm, pretrain_model = self.init_params()
        self.model = VaDE(n_neurons=n_neurons, batch_norm=batch_norm, k=k, device=device, landmark_files=landmark_files,
                          pretrained_model=pretrain_model, init_gmm=init_gmm, seqlen=seqlen, log_func=self.log_func)
        lr_gmm = ifnone(lr_gmm, lr)

    def prepare_data(self):
        landmark_datasets = {}
        for file in self.landmark_files:
            try:
                ds = LandmarkDataset(file)
                landmark_datasets[file] = ds
            except OSError:
                pass
        self.landmark_datasets = landmark_datasets
        self.landmark_files = list(self.landmark_datasets.keys())
        coords_dict = {file: sig.decimate(ds.coords, q=4, axis=0).astype(np.float32) for file, ds in landmark_datasets.items()}
        self.coords = [coords_dict[file] for file in self.landmark_files]
        N, n_coords, _ = self.coords[0].shape
        train_data = [crds[:int(0.8*crds.shape[0])].reshape(-1, n_coords*2) for crds in self.coords]
        valid_data = [crds[int(0.8*crds.shape[0]):].reshape(-1, n_coords*2) for crds in self.coords]
        train_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=3, diff=False) for data in train_data]
        valid_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=10, diff=False) for data in valid_data]
        self.train_ds = ConcatDataset(train_dsets)
        self.valid_ds = ConcatDataset(valid_dsets)
        all_data = {file: crds.reshape(-1, n_coords*2) for file, crds in coords_dict.items()}
        self.all_dsets_dict = {file: SequenceDataset(data, seqlen=self.seqlen, step=1, diff=False) for file, data in all_data.items()}
        self.all_ds = ConcatDataset([self.all_dsets_dict[file] for file in self.landmark_files])
        triplets_dataset = TripletsDataset(self.triplets_file, self.all_dsets_dict, seqlen=self.seqlen)
        train_idxs, valid_idxs = train_test_split(np.arange(len(triplets_dataset)), test_size=0.2,
                                                  random_state=self.hparams['triplets_random_state'])
        self.train_triplet_ds = CombinedDataset(self.train_ds, Subset(triplets_dataset, train_idxs))
        self.valid_triplet_ds = CombinedDataset(self.valid_ds, Subset(triplets_dataset, valid_idxs))

    def init_params(self):
        self.prepare_data()
        pretrain_model = SimpleAutoencoder(self.n_neurons, batch_norm=self.batch_norm, lr=3e-4)
        pretrain_model.val_dataloader = lambda: DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=12)
        pretrain_model.train_dataloader = lambda: DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=12)
        gpus = 1 if torch.cuda.is_available() else 0
        trainer = pl.Trainer(gpus=gpus, max_epochs=self.hparams['pretrain_epochs'], progress_bar_refresh_rate=10, logger=pl.loggers.WandbLogger('landmarks AE pretrain'))
        trainer.fit(pretrain_model)
        dataset = self.all_ds
        X_encoded = pretrain_model.encode_ds(dataset)
        init_gmm = GaussianMixture(self.hparams['k'], covariance_type='diag')
        init_gmm.fit(X_encoded)
        return (init_gmm, pretrain_model)
    
    # def prepare_data(self):
    #     pass
        # transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x).float()/255.)])
        # self.train_ds = MNIST("data", download=True)
        # self.valid_ds = MNIST("data", download=True, train=False)
        # to_tensor_dataset = lambda ds: TensorDataset(ds.data.view(-1, 28**2).float()/255., ds.targets)
        # self.train_ds, self.valid_ds = map(to_tensor_dataset, [self.train_ds, self.valid_ds])
        # self.all_ds = ConcatDataset([self.train_ds, self.valid_ds])
        # self.train_triplet_ds = CombinedDataset(MNIST("data", download=True), data_size=self.hparams['n_samples_for_triplets'])
        #                                         # transform=transforms.Lambda(lambda x: torch.flatten(x)/256))
        # self.valid_triplet_ds = CombinedDataset(MNIST("data", download=True, train=False), data_size=self.hparams['n_samples_for_triplets']) 
        #                                         # transform=transforms.Lambda(lambda x: torch.flatten(x)/256), seed=42)
            
    # def pretrain_model(self):
    #     n_neurons, pretrain_epochs, batch_norm = self.n_neurons, self.pretrain_epochs, self.batch_norm
    #     self.prepare_data()
    #     pretrained_model = SimpleAutoencoder(n_neurons, batch_norm=batch_norm, lr=3e-4)
    #     pretrained_model.val_dataloader = lambda: DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=8)
    #     pretrained_model.train_dataloader = lambda: DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=8)
    #     gpus = 1 if torch.cuda.is_available() else 0
    #     trainer = pl.Trainer(gpus=gpus, max_epochs=pretrain_epochs, progress_bar_refresh_rate=20)
    #     trainer.fit(pretrained_model)
    #     return pretrained_model
    
    # def init_params(self, k, pretrained_model=None):
    #     if not pretrained_model:
    #         pretrained_model = self.pretrain_model()
    #     transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: torch.flatten(x).float()/255.)])
    #     X_encoded = pretrained_model.encode_ds(self.all_ds)
    #     init_gmm = GaussianMixture(k, covariance_type='diag', n_init=5)
    #     init_gmm.fit(X_encoded)
    #     return pretrained_model, init_gmm
        
    def log_func(self, metric, value):
        self.log(self.state + '/' + metric, value)

    def train_dataloader(self):
        return DataLoader(self.train_triplet_ds, batch_size=self.batch_size, num_workers=8, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.valid_triplet_ds, batch_size=self.batch_size, num_workers=8)

    def configure_optimizers(self):
        # opt = torch.optim.AdamW([{'params': self.model.model_params},
        #                          {'params': self.model.gmm_params, 'lr': self.hparams['lr_gmm']}], 
        #                         self.hparams['lr'], weight_decay=0.00)
        # # sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: (epoch+1)/10 if epoch < 10 else 0.95**(epoch//10))
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"])
        lr_rate_function = lambda epoch: min((epoch+1)/self.hparams['warmup_epochs'], 0.9**(epoch//10))
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_rate_function)
        return [opt], [sched]

    def triplet_loss(self, triplets_batch):
        anchor_z, pos_z, neg_z = map(lambda s: self.model.encode(triplets_batch[s]).mean,
                                     ['anchor', 'positive', 'negative'])
        d1, d2 = torch.linalg.norm(anchor_z - pos_z, dim=1), torch.linalg.norm(anchor_z - neg_z, dim=1)
        self.log(self.state + '/anchor_pos_distance', d1.mean(), logger=True)
        self.log(self.state + '/anchor_neg_distance', d2.mean(), logger=True)
        self.log(self.state + '/correct_triplet_pct', (d1 < d2).float().mean()*100)
        anchor_z, pos_z, neg_z = map(lambda t: t / t.norm(dim=1, keepdim=True), [anchor_z, pos_z, neg_z])
        d1, d2 = torch.linalg.norm(anchor_z - pos_z, dim=1), torch.linalg.norm(anchor_z - neg_z, dim=1)
        self.log(self.state + '/normalized_anchor_pos_distance', d1.mean(), logger=True)
        self.log(self.state + '/normalized_anchor_neg_distance', d2.mean(), logger=True)
        self.log(self.state + '/normalized_correct_triplet_pct', (d1 < d2).float().mean()*100)
        loss = torch.relu(d1 - d2 + self.hparams['triplet_loss_margin']).mean()
        return loss

    def triplet_loss_kl(self, triplets_batch):
        anchor_z_dist, pos_z_dist, neg_z_dist = map(lambda s: self.model.encode(triplets_batch[s]), 
                                                    ['anchor', 'positive', 'negative'])
        d1, d2 = kl_distance(anchor_z_dist, pos_z_dist), kl_distance(anchor_z_dist, neg_z_dist)
        self.log(self.state + '/anchor_pos_distance_kl', d1.mean(), logger=True)
        self.log(self.state + '/anchor_neg_distance_kl', d2.mean(), logger=True)
        self.log(self.state + '/correct_triplet_pct_kl', (d1 < d2).float().mean()*100)
        loss = torch.relu(d1 - d2 + self.hparams['triplet_loss_margin_kl']).mean()
        return loss
    
    def shared_step(self, batch, batch_idx):
        bx, triplets_batch = batch
        result = self.model.shared_step(bx)
        result['triplet_loss'] = self.triplet_loss(triplets_batch)
        result['triplet_loss_kl'] = self.triplet_loss_kl(triplets_batch)
        result['main_loss'] = result['loss'].detach().clone()
        if self.hparams['triplet_loss_alpha'] > 0:
            result['loss'] += self.hparams['triplet_loss_alpha'] * result['triplet_loss']
        if self.hparams['triplet_loss_alpha_kl'] > 0:
            result['loss'] += self.hparams['triplet_loss_alpha_kl'] * result['triplet_loss_kl']
        # import pdb; pdb.set_trace()
        return result

    def validation_step(self, batch, batch_idx):
        self.state = 'valid'
        result = self.shared_step(batch, batch_idx)
        for k, v in result.items():
            self.log('valid/' + k, v, logger=True)
        return result

    def training_step(self, batch, batch_idx):
        self.state = 'train'
        result = self.shared_step(batch, batch_idx)
        for k, v in result.items():
            self.log('train/' + k, v, logger=True)
        return result

    def cluster_data(self, ds_type='all'):
        if ds_type=='all':
            dl = DataLoader(self.all_ds, batch_size=1024, shuffle=False, num_workers=8)
        elif ds_type=='train':
            dl = DataLoader(self.train_ds, batch_size=1024, shuffle=False, num_workers=8)
        elif ds_type=='valid':
            dl = DataLoader(self.valid_ds, batch_size=1024, shuffle=False, num_workers=8)
        else:
            raise Exception("Incorrect ds_type (can be one of 'train', 'valid', 'all')")
        return self.model.cluster_data(dl)




import pickle
data_root = Path('/mnt/storage2/shuki/data/THEMIS')

if not os.path.exists('deep_cluster/triplets/temp/vade_model.py'):
    landmark_files = list((data_root / 'landmarks').glob('0015*.h5'))
    print(landmark_files)
    vade_model = VaDE(landmark_files, 60, batch_norm=False, pretrain=False)
    vade_model.prepare_data()
    print("saving with pickle")
    import pickle
    os.makedirs('deep_cluster/triplets/temp/')
    with open('deep_cluster/triplets/temp/vade_model.py', 'wb') as file:
        pickle.dump(vade_model, file)
else:
    with open('deep_cluster/triplets/temp/vade_model.py', 'rb') as file:
        vade_model = pickle.load(file)


import wandb
SEED = 42

if __name__ == '__main__':
    triplets_file = 'deep_cluster/triplets/data/selected_triplets.csv'
    landmark_files = list((data_root / 'landmarks').glob('*.h5'))
    seqlen = 30
    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, triplets_random_state=SEED,
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file,
            triplet_loss_alpha=0., triplet_loss_alpha_kl=0., triplet_loss_margin_kl=1., batch_size=256, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()
    
    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, 
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file, triplets_random_state=SEED,
            triplet_loss_alpha=100, triplet_loss_alpha_kl=20, triplet_loss_margin_kl=1., batch_size=128, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()
    
    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, 
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file,
            triplet_loss_margin=0.3, triplets_random_state=SEED,
            triplet_loss_alpha=100, triplet_loss_alpha_kl=100, triplet_loss_margin_kl=0.5, batch_size=128, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()

    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, 
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file,
            triplet_loss_margin=0.5, triplets_random_state=SEED,
            triplet_loss_alpha=0.0, triplet_loss_alpha_kl=100, triplet_loss_margin_kl=1.0, batch_size=128, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()

    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, 
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file,
            triplet_loss_margin=0.5, triplets_random_state=SEED,
            triplet_loss_alpha=100, triplet_loss_alpha_kl=0.0, triplet_loss_margin_kl=1.0, batch_size=128, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()
    
    triplets_model = TripletVaDE(n_neurons=[2*seqlen*12, 512, 512, 30], k=30, 
            landmark_files=landmark_files, pretrain_epochs=30, triplets_file=triplets_file,
            triplet_loss_margin=1., triplets_random_state=SEED,
            triplet_loss_alpha=50, triplet_loss_alpha_kl=20, triplet_loss_margin_kl=3.0, batch_size=128, lr=3e-4)
    wandb.finish()
    logger = pl.loggers.WandbLogger(project='landmarks Triplet VADE')
    trainer = pl.Trainer(gpus=1, max_epochs=30, logger=logger)
    trainer.fit(triplets_model)
    wandb.finish()
    # triplets_ds = TripletsDataset(triplets_file, vade_model.all_dsets_dict, 30)
    
    # train_idxs, test_idxs = train_test_split(np.arange(len(triplets_ds)), test_size=0.2)
    # combined_dataset = CombinedDataset(vade_model.train_ds, Subset(triplets_ds, train_idxs))
    # print(triplets_ds.df.selected.value_counts())
    # print(triplets_ds[10])
    # print(combined_dataset.n_triplets)
