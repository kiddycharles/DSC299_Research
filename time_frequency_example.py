import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Generate synthetic time series data
np.random.seed(0)
n_samples = 1000
t = np.linspace(0, 10, n_samples)

# Time series with the same mean and variance but different frequency components
y1 = np.sin(2 * np.pi * t) + np.random.normal(0, 0.5, n_samples)
y2 = np.sin(2 * np.pi * 2 * t) + np.random.normal(0, 0.5, n_samples)

# Perform Fourier Transform
yf1 = fft(y1)
yf2 = fft(y2)
xf = fftfreq(n_samples, t[1] - t[0])

# Plot the time domain signals
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(t, y1)
plt.title('Time Domain Signal 1')
plt.subplot(2, 2, 2)
plt.plot(t, y2)
plt.title('Time Domain Signal 2')

# Plot the frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(xf, np.abs(yf1))
plt.title('Frequency Domain Signal 1')
plt.subplot(2, 2, 4)
plt.plot(xf, np.abs(yf2))
plt.title('Frequency Domain Signal 2')
plt.show()
