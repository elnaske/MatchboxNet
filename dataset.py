import torch
from torch import nn
from torchaudio.datasets import SPEECHCOMMANDS
from torchaudio.transforms import MFCC, TimeMasking, FrequencyMasking
from torch.utils.data import Dataset
from torchaudio import load

import os
from pathlib import Path
from glob import glob
from tqdm import tqdm
import random
import numpy as np

from data_transforms import PadTo, TimeShift, WhiteNoise, SpecCutout

def get_MFCC_transform():
    mfcc = MFCC(sample_rate=16000,
                n_mfcc=64,
                melkwargs={
                    'n_fft': 400,
                    'hop_length': 160,
                    'n_mels': 64
                })
    return mfcc

class Preprocessor(SPEECHCOMMANDS):
    def __init__(self, subset: str = None, version: int = 2, keywords: list = None, silence: bool = False):
        if version not in (1,2):
            raise ValueError('Version number must be 1 or 2.')

        super().__init__('./data', url = f'speech_commands_v0.0{version}', download=True)

        def load_list(filename):
            path = os.path.join(self._path, filename)
            with open(path) as f:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in f]

        if subset == 'validation':
            self._walker = load_list('validation_list.txt')
        elif subset == 'testing':
            self._walker = load_list('testing_list.txt')
        elif subset == 'training':
            exclude = set(load_list('validation_list.txt') + load_list('testing_list.txt'))
            self._walker = [w for w in self._walker if w not in exclude]

        # Preprocessing
        mfcc = get_MFCC_transform()
        # MFCC(sample_rate=16000,
        #         n_mfcc=64,
        #         melkwargs={
        #             'n_fft': 400,
        #             'hop_length': 160,
        #             'n_mels': 64
        #         })

        if subset == 'training':
            self.transform = nn.Sequential(
                TimeShift(80),
                WhiteNoise(46, 90),
                mfcc,
                PadTo(128)
            )
        else:
            self.transform = nn.Sequential(
                mfcc,
                PadTo(128)
            )

        self.label_mapper = LabelMapper(version, keywords, silence)

        self.label_counts = [0] * len(self.label_mapper.keywords)

        self.keywords = keywords


    def __getitem__(self, n):
        audio, _, label, *_ = super().__getitem__(n)

        if self.keywords is not None:
            if label not in self.keywords:
                label = 'UNK'

        X = self.transform(audio).squeeze()

        self.label_counts[self.label_mapper.label_to_idx(label)] += 1

        return X, label

class KeywordsDataset(Dataset):
    def __init__(self, partition, version=2, keywords: list = None, silence: bool = False):
        dataset = Preprocessor(partition, version=version, keywords=keywords, silence=silence)
        self.label_mapper = dataset.label_mapper
        self.transform = dataset.transform
        self.label_counts = dataset.label_counts
        self.n_classes = len(self.label_counts)

        self.labels = []
        self.X = []

        self.version = version
        self.keywords = keywords
        self.silence = silence

        self.preprocess(dataset)

    def preprocess(self, dataset):
        for i in tqdm(range(len(dataset)), 'Preprocessing Data'):
            X, label = dataset[i]

            self.X.append(X)
            self.labels.append(label)

        if self.silence:
            noise_samples = self.sample_noise()
            self.X += noise_samples
            self.labels += ['SIL'] * len(noise_samples)
            self.label_counts[-1] = len(noise_samples)

    def sample_noise(self):
        noise_files = glob(f'./data/SpeechCommands/speech_commands_v0.0{self.version}/_background_noise_/*.wav')
        n_noise_types = len(noise_files)
        n_silence = len(self.X) / (self.n_classes - 1)

        samples = []

        for f in noise_files:
            waveform, sr = load(f)

            for _ in range(int(n_silence // n_noise_types)):
                idx = random.randrange(0, waveform.shape[1] - sr)

                sample = waveform[:, idx:idx+sr]

                amplitude_modifier = random.randint(10,100) / 100
                sample *= amplitude_modifier

                samples.append(self.transform(sample).squeeze())

        return samples

    def get_class_weights(self):
        weights = torch.tensor([1 / n for n in self.label_counts])
        return weights / weights.sum() * len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = self.label_mapper.label_to_idx(self.labels[idx])

        return X, y

def collate_data_aug(batch):
    data_aug = nn.Sequential(
                    TimeMasking(25),
                    TimeMasking(25),
                    FrequencyMasking(15),
                    FrequencyMasking(15),
                    SpecCutout(15, 25, n_masks=5)
                )

    X, y = zip(*batch)

    X = data_aug(torch.stack(X))
    y = torch.stack(y)

    return X, y

class LabelMapper():
    def __init__(self, version=2, keywords=None, silence=False):
        labels = glob(f'./data/SpeechCommands/speech_commands_v0.0{version}/*/')
        labels = [Path(f).parts[-1] for f in labels if '_background_noise_' not in f]
        self.labels = sorted(list(set(labels)))

        if keywords is not None:
            self.keywords = keywords + ['UNK']
        else:
            self.keywords = self.labels

        if silence:
            self.keywords += ['SIL']
        self.silence = silence

        self.label_to_idx_map = {label: idx for idx, label in enumerate(self.keywords)}
        self.idx_to_label_map = {idx: label for label, idx in self.label_to_idx_map.items()}

    def label_to_idx(self, label):
        if type(label) == str:
            return torch.tensor(self.label_to_idx_map[label])
        elif type(label) == list:
            return torch.tensor([self.label_to_idx_map[l] for l in label])
        else:
            raise TypeError('Argument must be str or list of strings')

    def idx_to_label(self, idx):
        if isinstance(idx, int):
            return self.idx_to_label_map[idx]
        elif isinstance(idx, list):
            return [self.idx_to_label_map[i] for i in idx]
        elif isinstance(idx, torch.Tensor) or isinstance(idx, np.array):
            if idx.dim() == 0:
                return self.idx_to_label_map[idx.item()]
            elif idx.dim() == 1:
                return [self.idx_to_label_map[i.item()] for i in idx]
        else:
            raise TypeError('Argument must be int or iterable')
