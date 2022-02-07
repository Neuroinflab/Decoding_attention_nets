import random

import mne
import numpy as np


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
