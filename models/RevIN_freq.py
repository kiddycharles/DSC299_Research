import torch
import torch.nn as nn
import torch.fft


class RevIN_freq(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN_freq, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        # Convert to frequency domain
        self.x_freq = torch.fft.fft(x, dim=-1)
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(self.x_freq, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(self.x_freq, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()


    def _normalize(self, x):
        x = self.x_freq
        x = x - self.mean
        x = x / self.stdev
        x = torch.fft.ifft(x, dim=-1).real
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        # convert to frequency domain
        x_freq = torch.fft.fft(x, dim=-1)
        # Denormalize in frequency domain

        x_freq = x_freq * self.stdev
        x_freq = x_freq + self.mean

        # convert back to time domain
        x = torch.fft.ifft(x_freq, dim=-1).real
        return x
