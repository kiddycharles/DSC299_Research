import torch
import torch.nn as nn


class RevIN_w_SK(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN_w_SK, self).__init__()
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
        self.skewness_weight = nn.Parameter(torch.ones(self.num_features))
        self.kurtosis_weight = nn.Parameter(torch.ones(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
        # self.skewness = self._compute_skewness(x, dim2reduce)
        self.kurtosis = self._compute_kurtosis(x, dim2reduce)

    def _compute_skewness(self, x, dim2reduce):
        mean = self.mean
        stdev = self.stdev
        skewness = torch.mean(((x - mean) / stdev) ** 3, dim=dim2reduce, keepdim=True)
        return skewness

    def _compute_kurtosis(self, x, dim2reduce):
        mean = self.mean
        stdev = self.stdev
        kurtosis = torch.mean(((x - mean) / stdev) ** 4, dim=dim2reduce, keepdim=True) - 3
        return kurtosis

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev

        # Adjust based on skewness and kurtosis
        # self.skewness_adjustment = (self.skewness_weight + self.eps) / (self.skewness + self.eps)
        self.kurtosis_adjustment = (self.kurtosis_weight + self.eps) / (self.kurtosis + self.eps)

        # x = x * self.skewness_adjustment
        x = x / self.kurtosis_adjustment

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.kurtosis_adjustment
        # x = x / self.skewness_adjustment
        x = x * self.stdev
        x = x + self.mean
        return x
