import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.fft

# Generate synthetic time series data
np.random.seed(0)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
y1 = np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n_samples)
y2 = np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.5, n_samples)

# Convert data to torch tensors
y1 = torch.tensor(y1, dtype=torch.float32)
y2 = torch.tensor(y2, dtype=torch.float32)


# Function to perform joint time and frequency domain normalization
def hybrid_normalization(y):
    # Time domain normalization
    time_mean = torch.mean(y)
    time_std = torch.std(y)
    y_time_normalized = (y - time_mean) / time_std

    # Frequency domain normalization
    yf = torch.fft.fft(y_time_normalized)
    amplitude = torch.abs(yf)
    phase = torch.angle(yf)
    freq_mean_amplitude = torch.mean(amplitude)
    freq_std_amplitude = torch.std(amplitude)
    normalized_amplitude = (amplitude - freq_mean_amplitude) / freq_std_amplitude

    # Recombine normalized amplitude with original phase
    yf_normalized = normalized_amplitude * torch.exp(1j * phase)

    # Inverse Fourier Transform to get back to time domain
    y_normalized = torch.fft.ifft(yf_normalized).real

    # Final adjustment to match the original data's mean and variance
    final_mean = torch.mean(y)
    final_std = torch.std(y)
    y_normalized = y_normalized * final_std + final_mean

    return y_normalized, time_mean, time_std, freq_mean_amplitude, freq_std_amplitude


# Apply hybrid normalization
y1_normalized, y1_time_mean, y1_time_std, y1_freq_mean, y1_freq_std = hybrid_normalization(y1)
y2_normalized, y2_time_mean, y2_time_std, y2_freq_mean, y2_freq_std = hybrid_normalization(y2)

# Convert the normalized time series back to numpy for plotting
y1_normalized = y1_normalized.numpy()
y2_normalized = y2_normalized.numpy()

# Plot the original and normalized time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, y1.numpy())
plt.title('Original Time Series 1')
plt.subplot(2, 2, 2)
plt.plot(t, y2.numpy())
plt.title('Original Time Series 2')

plt.subplot(2, 2, 3)
plt.plot(t, y1_normalized)
plt.title('Normalized Time Series 1')
plt.subplot(2, 2, 4)
plt.plot(t, y2_normalized)
plt.title('Normalized Time Series 2')
plt.show()
