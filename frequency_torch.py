import torch

# Define a time-domain signal
x = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)

# Compute the FFT
X = torch.fft.fft(x)
print("FFT:", X)

# Compute the Inverse FFT
x_reconstructed = torch.fft.ifft(X)
print("Reconstructed Signal:", x_reconstructed)


def _get_statistics(x):
    X = torch.fft.fft(x)
    dim2reduce = tuple(range(1, x.ndim - 1))
    mean = torch.mean(X, dim=dim2reduce, keepdim=True).detach()
    stdev = torch.sqrt(torch.var(X, dim=dim2reduce, keepdim=True, unbiased=False) + 1e-8).detach()
    return mean, stdev


print(_get_statistics(x))