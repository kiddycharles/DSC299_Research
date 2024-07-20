import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Generate a sample signal: a sum of two sinusoids
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector
f1, f2 = 50, 150  # Frequencies of the sinusoids
signal = 2 * np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(1 * np.pi * f2 * t)

# Normalize in time domain
normalized_signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# FFT of the original and normalized signal
original_fft = fft(signal)
normalized_fft = fft(normalized_signal)

# Frequencies corresponding to FFT
frequencies = np.fft.fftfreq(len(t), 1/fs)

# Plot time domain signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.subplot(2, 2, 2)
plt.plot(t, normalized_signal)
plt.title('Normalized Signal (Time Domain)')

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(frequencies, np.abs(original_fft))
plt.title('Original Signal FFT')
plt.xlim(0, 200)  # Focus on lower frequencies
plt.subplot(2, 2, 4)
plt.plot(frequencies, np.abs(normalized_fft))
plt.title('Normalized Signal FFT')
plt.xlim(0, 200)  # Focus on lower frequencies

plt.tight_layout()
plt.show()
