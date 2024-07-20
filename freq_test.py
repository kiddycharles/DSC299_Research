import torch

# Example tensor with shape (batch_size, channels, seq_len)
x = torch.randn(32, 10, 100)  # e.g., batch_size=32, channels=10, seq_len=100

# Apply FFT along the seq_len dimension
x_fft = torch.fft.fft(x, dim=1)

# Compute magnitude of the FFT (use absolute value)
x_fft_mag = torch.abs(x_fft)

# Compute mean of the magnitude along the seq_len dimension
mean_fft = x_fft_mag.mean(dim=2)

# Compute variance of the magnitude along the seq_len dimension
variance_fft = x_fft_mag.var(dim=2, unbiased=False)

# Print shapes to confirm
print(f"Mean in frequency domain shape: {mean_fft.shape}")  # Should be (batch_size, channels)
print(f"Variance in frequency domain shape: {variance_fft.shape}")  # Should be (batch_size, channels)


def _get_statistics(x_in):
    dim2reduce = tuple(range(1, x_in.ndim - 1))
    mean = torch.mean(x_in, dim=dim2reduce, keepdim=True)
    stdev = torch.sqrt(torch.var(x_in, dim=dim2reduce, keepdim=True, unbiased=False))
    return mean, stdev


mean, stdev = _get_statistics(x)
print(f"Mean in time domain shape: {mean.shape}")
print(f"Variance in time domain shape: {stdev.shape}")