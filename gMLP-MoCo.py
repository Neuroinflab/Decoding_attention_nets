import random

import mne.filter
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F

from gMLP import gMLPVision


class EP(pl.LightningModule):

    def __init__(self, e_dim, p_emb_dim, depth, patch_size):
        super().__init__()

        self.encoder = gMLPVision(
            image_size=1280,
            patch_size=patch_size,
            dim=e_dim,
            depth=depth,
            ff_mult=6,
            channels=19,
            attn_dim=64,
            causal=False
        )
        self.proj = Projector(e_dim, e_dim, p_emb_dim)

    def forward(self, x):
        z = self.encoder(x)
        z = self.proj(z)

        return z


class Projector(pl.LightningModule):
    def __init__(self, input_dim=2048, hidden_dim=128, output_dim=32):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.model(x)

        return z


class MocoV2(pl.LightningModule):

    def __init__(self,
                 e_dim: int = 128,
                 p_emb_dim: int = 64,
                 depth: int = 30,
                 patch_size: int = 32,
                 num_negatives: int = 2**15,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.5,
                 learning_rate: float = 1e-4,
                 *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        encoder_projection_params = {
            'e_dim': self.hparams.e_dim,
            'p_emb_dim': self.hparams.p_emb_dim,
            'depth': self.hparams.depth,
            'patch_size': self.hparams.patch_size
        }
        # Init encoders
        self.encoder_q = EP(**encoder_projection_params)
        self.encoder_k = EP(**encoder_projection_params)

        # Turn off gradients for key encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(self.hparams.p_emb_dim, self.hparams.num_negatives))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, img_q, img_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self.encoder_q(img_q)  # queries: NxC
        q = F.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k = self.encoder_k(img_k)  # keys: NxC
            k = F.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def training_step(self, batch, batch_idx):
        img_1, img_2 = batch

        output, target = self(img_q=img_1, img_k=img_2)
        loss = F.cross_entropy(output.float(), target.long())

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.log('train_loss', loss.item(), prog_bar=True)
        self.log('train_acc1', acc1, prog_bar=True)
        self.log('train_acc5', acc5, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)

        return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TwoCropsTransform():
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


transforms = []


def transform(min_value, max_value):
    def decorate(func):
        transforms.append((func, min_value, max_value))

    return decorate


class TimeShift():
    def __init__(self, m, sampling_rate=256.0):
        self.m = m
        self.sampling_rate = sampling_rate

    def __call__(self, sample):
        sr = int(self.sampling_rate) // 2

        if self.m == 0:
            start = sr
        else:
            start = np.random.randint(sr - sr / 30 * self.m, sr + sr / 30 * self.m)

        return sample[:, start:start + 5 * int(self.sampling_rate)]


@transform(0.00, 4.00)
class Cutout():
    def __init__(self, duration, sampling_rate=256.0):
        self.duration = duration
        self.sampling_rate = sampling_rate

    def __call__(self, sample):
        sample_copy = sample.copy()

        start = np.random.randint(0, sample.shape[1] - self.sampling_rate * self.duration)
        sample_copy[:, start:start + int(self.sampling_rate * self.duration)] = 0

        return sample_copy


@transform(0, 1)
class Identity():
    def __init__(self, foo):
        pass

    def __call__(self, sample):
        return sample


@transform(0.0, 10.0)
class Jittering():
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        if self.sigma == 0.0:
            return sample

        noise = np.random.normal(0, self.sigma, size=sample.shape)

        return sample + noise


@transform(0.0, 0.5)
class SensorDropout():
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return (x * np.random.binomial(1, 1 - self.p, 19)[:, np.newaxis])


@transform(0.0, 1.0)
class Scaling:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, sample):
        factor = np.random.normal(1.0, self.sigma, size=(19, 1))
        return sample * factor


@transform(0.0, 1.0)
class FlippingDropout():
    def __init__(self, p):
        self.p = p

    def __call__(self, x):
        return (x * ((np.random.binomial(1, 1 - self.p, 19) - 1 / 2) * 2).astype(int)[:, np.newaxis])


@transform(0.0, 30.0)
class DCShift():
    def __init__(self, r):
        self.min_offset = 20 * (r / 30)
        self.max_offset = 50 * (r / 30)

    def __call__(self, x):
        sign = random.choice([-1, 1])
        value = np.random.uniform(self.min_offset, self.max_offset)

        return x + (sign * value)


@transform(1.0, 24.0)
class Bandstop():
    def __init__(self, width):
        self.width = width

    def __call__(self, x):
        freq = np.random.uniform(1.0, 40.0)
        lfreq = freq - self.width / 2
        hfreq = freq + self.width / 2

        if lfreq < 1.0:
            correction = 1.0 - lfreq
        elif hfreq > 40.0:
            correction = 40.0 - hfreq
        else:
            correction = 0.0

        lfreq += correction
        hfreq += correction

        with mne.utils.use_log_level('error'):
            return mne.filter.filter_data(x.astype(np.float64), sfreq=256.0, l_freq=hfreq, h_freq=lfreq,
                                          method='iir').astype(np.float32)


class RandAug():
    def __init__(self, n, m):
        self.n = n
        self.m = m

    def __call__(self, sample):
        if self.n == 0:
            return sample

        ops = random.choices(transforms, k=self.n)
        for op, min_val, max_val in ops:
            val = (self.m / 30) * (max_val - min_val) + min_val
            sample = op(val)(sample)

        return sample.astype(np.float32)
