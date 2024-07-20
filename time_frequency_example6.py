import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Generate two signals with the same mean and variance but different frequency content
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector

# Signal 1: Sum of two different frequencies
f1, f2 = 50, 150  # Frequencies of the sinusoids
signal1 = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Signal 2: Different frequencies and amplitudes but same mean and variance
f3, f4 = 75, 175  # Different frequencies
signal2 = np.sin(2 * np.pi * f3 * t) + 0.5 * np.sin(2 * np.pi * f4 * t)

print("Signal 1 mean:", signal1.mean())
print("Signal 1 variance:", signal1.var())

print("Signal 2 mean:", signal2.mean())
print("Signal 2 variance:", signal2.var())


# Normalize both signals in time domain
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)


normalized_signal1 = normalize(signal1)
normalized_signal2 = normalize(signal2)

print("Signal 1 mean (after normalization):", normalized_signal1.mean())
print("Signal 1 variance (after normalization):", normalized_signal1.var())

print("Signal 2 mean (after normalization):", normalized_signal2.mean())
print("Signal 2 variance (after normalization):", normalized_signal2.var())

# Compute FFT for both signals
fft_signal1 = fft(normalized_signal1)
fft_signal2 = fft(normalized_signal2)

# Frequencies corresponding to FFT
frequencies = np.fft.fftfreq(len(t), 1 / fs)

# Plot time domain signals
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(t, normalized_signal1)
plt.title('Normalized Signal 1 (Time Domain)')
plt.ylim([-3, 3])

plt.subplot(2, 2, 2)
plt.plot(t, normalized_signal2)
plt.title('Normalized Signal 2 (Time Domain)')
plt.ylim([-3, 3])

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(frequencies, np.abs(fft_signal1))
plt.title('Normalized Signal 1 FFT')
plt.xlim(0, 200)  # Focus on lower frequencies

plt.subplot(2, 2, 4)
plt.plot(frequencies, np.abs(fft_signal2))
plt.title('Normalized Signal 2 FFT')
plt.xlim(0, 200)  # Focus on lower frequencies

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Generate two signals with the same mean and variance but different frequency content
fs = 1000  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)  # Time vector

# Signal 1: Combination of low and high frequency components
f1, f2 = 50, 300  # Frequencies of the sinusoids
signal1 = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Signal 2: Different frequencies and amplitudes but same mean and variance
f3, f4 = 100, 400  # Different frequencies
signal2 = np.sin(2 * np.pi * f3 * t) + 0.5 * np.sin(2 * np.pi * f4 * t)


print("Signal 1 mean:", signal1.mean())
print("Signal 1 variance:", signal1.var())

print("Signal 2 mean:", signal2.mean())
print("Signal 2 variance:", signal2.var())


# Normalize both signals in time domain
def normalize(signal):
    return (signal - np.mean(signal)) / np.std(signal)

normalized_signal1 = normalize(signal1)
normalized_signal2 = normalize(signal2)

# Compute FFT for both signals
fft_signal1 = fft(normalized_signal1)
fft_signal2 = fft(normalized_signal2)

# Frequencies corresponding to FFT
frequencies = np.fft.fftfreq(len(t), 1/fs)

# Plot time domain signals
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(t, normalized_signal1)
plt.title('Normalized Signal 1 (Time Domain)')
plt.ylim([-3, 3])

plt.subplot(2, 2, 2)
plt.plot(t, normalized_signal2)
plt.title('Normalized Signal 2 (Time Domain)')
plt.ylim([-3, 3])

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(frequencies, np.abs(fft_signal1))
plt.title('Normalized Signal 1 FFT')
plt.xlim(0, 500)  # Focus on lower frequencies

plt.subplot(2, 2, 4)
plt.plot(frequencies, np.abs(fft_signal2))
plt.title('Normalized Signal 2 FFT')
plt.xlim(0, 500)  # Focus on lower frequencies

plt.tight_layout()
plt.show()
