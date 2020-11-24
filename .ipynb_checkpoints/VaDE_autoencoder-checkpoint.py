import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from scipy import signal as sig
import torch
from dataloader import LandmarkDataset, SequenceDataset
from torch import autograd
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch import nn
from torch.nn import functional as F
from torch import distributions as D
import pytorch_lightning as pl
from matplotlib import pyplot as plt
from torch.distributions import Normal, Laplace, kl_divergence, kl, Categorical
import math
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import metrics




def get_autoencoder(n_neurons, batch_norm=True):
    enc_layers = len(n_neurons)
    layer_dims = n_neurons + n_neurons[-2::-1]
    n_layers = len(layer_dims)
    return nn.Sequential(*[nn.Sequential(nn.Linear(layer_dims[i], layer_dims[i+1]),
                                         nn.Identity() \
                                            if i+2 == enc_layers or i+2 == n_layers or not batch_norm \
                                            else nn.BatchNorm1d(layer_dims[i+1]),
                                         nn.Identity() \
                                            if i+2 == enc_layers or i+2 == n_layers \
                                            else nn.ELU()) \
                           for i in range(n_layers - 1)])

def get_encoder_decoder(n_neurons, batch_norm=True):
    n_layers = len(n_neurons) - 1
    encoder_layers = [nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                    nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity(),
                                    nn.ELU()) for i in range(n_layers - 1)]
    encoder_layers.append(nn.Linear(n_neurons[-2], n_neurons[-1]))
    n_neurons = n_neurons[::-1]
    decoder_layers = [nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                                    nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity(),
                                    nn.ELU()) for i in range(n_layers - 1)]
    decoder_layers.append(nn.Linear(n_neurons[-2], n_neurons[-1])) 
    return nn.Sequential(*encoder_layers), nn.Sequential(*decoder_layers)


class SimpleAutoencoder(pl.LightningModule):
    def __init__(self, n_neurons=[784, 512, 256, 10], lr=1e-3, batch_norm=True):
        super(SimpleAutoencoder, self).__init__()
        self.hparams = {'lr': lr}
        # self.model = get_autoencoder(n_neurons, batch_norm)
        self.encoder, self.decoder = get_encoder_decoder(n_neurons, batch_norm)
        # self.encoder = self.model[:len(n_neurons) - 1]
        # self.decoder = self.model[len(n_neurons) - 1:]

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.hparams['lr'])
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.2 ,patience=20, verbose=True, min_lr=1e-6)
        scheduler = {'scheduler': sched, 'interval': 'epoch', 'monitor': 'val_checkpoint_on', 'reduce_on_plateau': True}
        return opt

    def shared_step(self, batch, batch_idx):
        bx = batch
        z = self.encoder(bx)
        out = self.decoder(z)
        loss = F.mse_loss(out, bx)
#         bce_loss = F.binary_cross_entropy(torch.clamp(out, 1e-6, 1 - 1e-6), bx, reduction='mean')
        lmbda = 0.00
        reg = (z**2).mean()
        self.log('loss', loss)
        self.log('regularization', reg)
#         self.log('bce_loss', bce_loss)
        return loss + lmbda * reg
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)

    def encode(self, batch):
        bx = batch
        return self.encoder(bx)

    def encode_ds(self, ds):
        dl = DataLoader(ds, batch_size=1024, num_workers=4, shuffle=False)
        encoded = []
        with torch.no_grad():
            for batch in dl:
                encoded.append(self.encode(batch).detach().cpu().numpy())
        return np.concatenate(encoded, axis=0)

    def cluster(self, ds, k=10):
        X = self.encode_ds(ds)
        kmeans = KMeans(k)
        return kmeans.fit_predict(X)

    
def cross_entropy(P, Q):
    return kl_divergence(P, Q) + P.entropy()


class LatentDistribution(nn.Module):
    prior = Normal(0, 1)
    eps = 1e-8
    def __init__(self, in_features, out_features, sigma=None, same_sigma=False, distribution='normal'):
        super(LatentDistribution, self).__init__()
        self.distribution = distribution
        self.mu_fc = nn.Linear(in_features, out_features)
        if sigma:
            self.sigma = sigma
        else:
            if same_sigma:
                self.logvar_fc = nn.Linear(in_features, 1)
                self.logvar_fc.weight.data.zero_()
                self.logvar_fc.bias.data.zero_()
            else:
                self.logvar_fc = nn.Linear(in_features, out_features)
        
    
    def forward(self, x):
        mu = self.mu_fc(x)
        if hasattr(self, 'sigma'):
            sigma = self.sigma
        else:
            logvar = self.logvar_fc(x)
            logvar = torch.clamp(logvar, min=-12, max=8)
            sigma = torch.exp(logvar / 2) + self.eps
        if self.distribution == 'normal':
            self.dist = Normal(mu, sigma)
        elif self.distribution == 'cauchy':
            self.dist = D.Cauchy(loc=mu, scale=sigma)
        return self.dist
    
    def sample(self, l=1):
        return self.dist.rsample()

    def kl_loss(self, prior=None):
        if not prior:
            prior = self.prior
        return kl_divergence(self.dist, prior).sum(dim=-1)
    
    
def xlogx(x, eps=1e-12):
    xlog = x * (x + eps).log()
    return xlog


def clustering_accuracy(gt, labels):
    mat = metrics.confusion_matrix(labels, gt)
    labels = mat.argmax(axis=1)[labels]
    return metrics.accuracy_score(gt, labels)



class VaDE(pl.LightningModule):
    def __init__(self, landmark_files, seqlen, n_neurons=[784, 512, 256, 10], batch_norm=True, k=10, lr=1e-3, device='cuda', pretrain=True):
        super(VaDE, self).__init__()
        self.k = k
        self.seqlen = seqlen
        self.landmark_files = landmark_files
        self.n_neurons, self.batch_norm = n_neurons, batch_norm
        self.hparams = {'lr': lr}
        self.latent_dim = n_neurons[-1]
        self.mixture_logits = nn.Parameter(torch.zeros(k, device=device))
        self.mu_c = nn.Parameter(torch.zeros(self.latent_dim, k, device=device))
        self.logvar_c = nn.Parameter(torch.zeros(self.latent_dim, k, device=device))
#         self.sigma_c = nn.Parameter(torch.ones(self.latent_dim, k, device=device))
        n_layers = len(n_neurons) - 1
        layers = list()
        for i in range(n_layers-1):
            layers.append(nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                          nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity(),
                          nn.ELU()))
        self.encoder = nn.Sequential(*layers)
        self.latent_dist = LatentDistribution(n_neurons[-2], n_neurons[-1])
        
        layers = list()
        n_neurons = n_neurons[::-1]
        for i in range(n_layers-1):
            layers.append(nn.Sequential(nn.Linear(n_neurons[i], n_neurons[i+1]),
                          nn.BatchNorm1d(n_neurons[i+1]) if batch_norm else nn.Identity(),
                          nn.ELU()))
        self.decoder = nn.Sequential(*layers)
        self.sigma = nn.Parameter(torch.ones(1)*np.sqrt(0.02), requires_grad=False)
        # self.sigma = 1
        self.out_dist = LatentDistribution(n_neurons[-2], n_neurons[-1], sigma=self.sigma, distribution='cauchy')
        if pretrain:
            self.init_params()


    def init_params(self):
        self.prepare_data()
        pretrain_model = SimpleAutoencoder(self.n_neurons, batch_norm=self.batch_norm, lr=1e-3)
        pretrain_model.val_dataloader = self.val_dataloader
        pretrain_model.train_dataloader = self.train_dataloader
        trainer = pl.Trainer(gpus=1, max_epochs=15, progress_bar_refresh_rate=10, logger=pl.loggers.WandbLogger('AE pretrain'))
        trainer.fit(pretrain_model)
        dataset = self.all_ds
        X_encoded = pretrain_model.encode_ds(dataset)
        init_gmm = GaussianMixture(self.k, covariance_type='diag')
        init_gmm.fit(X_encoded)
        self.mixture_logits.data = torch.Tensor(np.log(init_gmm.weights_))
        self.mu_c.data = torch.Tensor(init_gmm.means_.T)
        self.logvar_c.data = torch.Tensor(init_gmm.covariances_.T).log()
#         self.sigma_c.data = torch.Tensor(init_gmm.covariances_.T).sqrt()
        self.encoder.load_state_dict(pretrain_model.encoder[:-1].state_dict())
        self.decoder.load_state_dict(pretrain_model.decoder[:-1].state_dict())
        self.latent_dist.mu_fc.load_state_dict(pretrain_model.encoder[-1].state_dict())
        self.out_dist.mu_fc.load_state_dict(pretrain_model.decoder[-1].state_dict())

    def forward(self, bx):
        x = self.encoder(bx)
        z_dist = self.latent_dist(x)
        z = z_dist.rsample()
        x_dist = self.out_dist(self.decoder(z))
        return x_dist

    def shared_step(self, bx):
        x = self.encoder(bx)
        z_dist = self.latent_dist(x)
        if z_dist.scale.max() > 100:
            import pdb; pdb.set_trace()
        if z_dist.scale.log().min() < -20:
            import pdb; pdb.set_trace()
        self.log("latent dist std", z_dist.scale.mean().detach())
        self.log("mixture logits std", self.mixture_logits.std())
        mixture_probs = self.mixture_logits.softmax(0).detach()
        self.log("mixture distrioution entropy", - torch.sum(mixture_probs * mixture_probs.log()))
        z = z_dist.rsample()
        x_dist = self.out_dist(self.decoder(z))
        x_recon_loss = - x_dist.log_prob(bx).sum(dim=-1)
        mse_loss = F.mse_loss(x_dist.loc, bx)
        
        ###################################
        
        self.sigma_c = torch.exp(self.logvar_c / 2)
        comp_dists = [D.Normal(self.mu_c[:,i], self.sigma_c[:,i]) for i in range(self.k)]
        gmm = D.MixtureSameFamily(D.Categorical(logits=self.mixture_logits), D.Normal(self.mu_c, self.sigma_c))
        
        log_p_z_c = gmm.component_distribution.log_prob(z.unsqueeze(-1)).sum(dim=1)
        q_c_z = torch.softmax(log_p_z_c + (gmm.mixture_distribution.probs + 1e-9).log() , dim=-1)  # dims: (bs, k)
        cross_entropies = torch.stack([cross_entropy(z_dist, comp_dists[i]).sum(dim=1) for i in range(self.k)], dim=-1)
        crosent_loss1 = - (cross_entropies * q_c_z).sum(dim=-1)
        crosent_loss2 = (q_c_z * (gmm.mixture_distribution.probs[None] + 1e-9).log()).sum(dim=-1)
        ent_loss1 = z_dist.entropy().sum(dim=-1)
        ent_loss2 = - xlogx(q_c_z).sum(dim=-1)
        
        self.log('ent_loss1', - ent_loss1.mean())
        self.log('ent_loss2', - ent_loss2.mean())
        self.log('crosent_loss1', - crosent_loss1.mean())
        self.log('crosent_loss2', - crosent_loss2.mean())
        self.log('log likelihood', gmm.log_prob(z).mean())

        ############################################################################
        kl_loss = - crosent_loss1 - crosent_loss2 - ent_loss1 - ent_loss2
        loss = x_recon_loss + kl_loss
        if not torch.all(loss.isfinite()):
            import pdb; pdb.set_trace()
        return loss.mean(), x_recon_loss.mean(), kl_loss.mean(), mse_loss.mean()

    def prepare_data(self):
        landmark_datasets = []
        for file in self.landmark_files:
            try:
                ds = LandmarkDataset(file)
                landmark_datasets.append(ds)
            except OSError:
                pass
        self.landmark_datasets = landmark_datasets
        coords = [sig.decimate(ds.coords, q=4, axis=0).astype(np.float32) for ds in landmark_datasets]
        self.coords = coords
        N, n_coords, _ = coords[0].shape
        train_data = [crds[:int(0.8*crds.shape[0])].reshape(-1, n_coords*2) for crds in coords]
        valid_data = [crds[int(0.8*crds.shape[0]):].reshape(-1, n_coords*2) for crds in coords]
        train_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=1, diff=False) for data in train_data]
        valid_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=20, diff=False) for data in valid_data]
        self.train_ds = ConcatDataset(train_dsets)
        self.valid_ds = ConcatDataset(valid_dsets)
        all_data = [crds.reshape(-1, n_coords*2) for crds in coords]
        all_dsets = [SequenceDataset(data, seqlen=self.seqlen, step=1, diff=False) for data in all_data]
        self.all_ds = ConcatDataset(all_dsets)
        
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=256, shuffle=True, num_workers=8)

    def val_dataloader(self):
        # dataset = SequenceDataset(X_val, seqlen=30, step=5, diff=True)
        return DataLoader(self.valid_ds, batch_size=256, shuffle=False, num_workers=8)

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), self.hparams['lr'], weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: (epoch+1)/10 if epoch < 10 else 0.9**(epoch//10))
        return [opt], [sched]
    
    def training_step(self, batch, batch_idx):
        bx = batch
        loss, rec_loss, kl_loss, mse_loss = self.shared_step(bx)
        result = {'loss': loss, 
                  'rec_loss': rec_loss.detach(),
                  'kl_loss': kl_loss.detach(),
                  'mse_loss': mse_loss.detach()}
        for k, v in result.items():
            self.log(k, v, on_step=True)
        return result

    def validation_step(self, batch, batch_idx):
        bx = batch
        loss, rec_loss, kl_loss, mse_loss = self.shared_step(bx)
        result = {'loss': loss.detach(), 
                  'rec_loss': rec_loss.detach(),
                  'kl_loss': kl_loss.detach(),
                  'mse_loss': mse_loss.detach()}
        return result

    def cluster_data(self, dl=None):
        if not dl:
            dl = self.val_dataloader()
        self.sigma_c = torch.exp(self.logvar_c / 2)
        vade_gmm = D.MixtureSameFamily(D.Categorical(logits=self.mixture_logits), D.Normal(self.mu_c, self.sigma_c))
        labels = []
        X_encoded = []
        with torch.no_grad():
            for bx in dl:
                x_encoded = self.latent_dist(self.encoder(bx.cuda())).loc
                X_encoded.append(x_encoded)
                log_p_z_given_c = vade_gmm.component_distribution.log_prob(x_encoded.unsqueeze(2)).sum(dim=1)
                labels.append((log_p_z_given_c + vade_gmm.mixture_distribution.logits).softmax(dim=-1).argmax(dim=-1))
                
        labels = torch.cat(labels).cpu().numpy()
        X_encoded = torch.cat(X_encoded).cpu().numpy()
        return labels, X_encoded

