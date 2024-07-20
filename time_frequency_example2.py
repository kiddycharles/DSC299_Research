import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# Generate synthetic time series data
np.random.seed(0)
n_samples = 1000
t = np.linspace(0, 10, n_samples)
y1 = np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n_samples)
y2 = np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.5, n_samples)

# Function to normalize in the time domain
def time_domain_normalization(y):
    mean = np.mean(y)
    std = np.std(y)
    return (y - mean) / std, mean, std

# Function to normalize in the frequency domain
def frequency_domain_normalization(y):
    yf = fft(y)
    xf = fftfreq(n_samples, t[1] - t[0])
    amplitude = np.abs(yf)
    phase = np.angle(yf)
    mean_amplitude = np.mean(amplitude)
    std_amplitude = np.std(amplitude)
    normalized_amplitude = (amplitude - mean_amplitude) / std_amplitude
    return normalized_amplitude * np.exp(1j * phase), mean_amplitude, std_amplitude

# Normalize in the time domain
y1_time_normalized, y1_mean, y1_std = time_domain_normalization(y1)
y2_time_normalized, y2_mean, y2_std = time_domain_normalization(y2)

# Normalize in the frequency domain
y1_freq_normalized, y1_amp_mean, y1_amp_std = frequency_domain_normalization(y1_time_normalized)
y2_freq_normalized, y2_amp_mean, y2_amp_std = frequency_domain_normalization(y2_time_normalized)

# Inverse Fourier Transform to get back to time domain
y1_normalized = ifft(y1_freq_normalized).real
y2_normalized = ifft(y2_freq_normalized).real

# Plot the original and normalized time series
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, y1)
plt.title('Original Time Series 1')
plt.subplot(2, 2, 2)
plt.plot(t, y2)
plt.title('Original Time Series 2')

plt.subplot(2, 2, 3)
plt.plot(t, y1_normalized)
plt.title('Normalized Time Series 1')
plt.subplot(2, 2, 4)
plt.plot(t, y2_normalized)
plt.title('Normalized Time Series 2')
plt.show()
