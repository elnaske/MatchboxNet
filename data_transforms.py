import torch
from torch import nn
import torch.nn.functional as F
import random

class PadTo(nn.Module):
    def __init__(self, pad_to):
        super().__init__()

        self.pad_to = pad_to

    def forward(self, x):
        pad_amount = self.pad_to - x.shape[-1]

        pad_l = pad_amount // 2
        pad_r = pad_amount - pad_l

        return F.pad(x, (pad_l, pad_r))


class TimeShift(nn.Module):
    def __init__(self, shift_max):
        super().__init__()

        self.shift_max = shift_max

    def forward(self, x):
        shift = random.randrange(-self.shift_max, self.shift_max)

        if shift != 0:
            x = torch.roll(x, shift, dims=1)

            if shift > 0:
                x[:, :shift] = 0
            else:
                x[:, shift:] = 0

        return x

class WhiteNoise(nn.Module):
    def __init__(self, min_dB, max_dB):
        super().__init__()

        self.min_db = min_dB
        self.max_db = max_dB

    def forward(self, x):
        # SNR (in dB) = 10 * log10(P_signal / P_noise)
        # thus:
        # P_noise = P_signal / (10^(SNR / 10))

        P_signal = x.pow(2).mean()

        target_snr = random.uniform(self.min_db, self.max_db)

        denom = 10 ** (target_snr / 10)
        P_noise = P_signal / denom

        noise = torch.randn_like(x) * torch.sqrt(P_noise)

        return x + noise


class SpecCutout(nn.Module):
    def __init__(self, max_height, max_width, n_masks=1):
        super().__init__()

        self.max_height = max_height
        self.max_width = max_width
        self.n_masks = n_masks

    def forward(self, x):
        n_samples = x.shape[0]
        sample_height = x.shape[1]
        sample_width = x.shape[2]

        for _ in range(self.n_masks):
            height = random.randint(0, self.max_height)
            width = random.randint(0, self.max_width)

            freq_start = random.randrange(sample_height - height)
            freq_end = freq_start + height

            time_start = random.randrange(sample_width - width)
            time_end = time_start + width

            x[:, freq_start:freq_end, time_start:time_end] = 0

        return x