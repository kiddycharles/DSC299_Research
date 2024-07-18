import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 1000  # number of points
t = np.linspace(0, 10, n)

# Window 1: Sum of two sine waves
f1 = 1.5  # frequency 1
f2 = 4.0  # frequency 2
window1 = np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)

# Window 2: Chirp signal
f0 = 1.0  # starting frequency
f1 = 5.0  # ending frequency
window2 = np.sin(2 * np.pi * (f0 + (f1 - f0) * t / 10) * t)


# Normalize both windows to have the same mean and variance
def normalize(x):
    return (x - np.mean(x)) / np.std(x)


window1 = normalize(window1)
window2 = normalize(window2)

# Time domain plot
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.plot(t, window1, label='Window 1')
plt.plot(t, window2, label='Window 2')
plt.title('Time Domain')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()


# Frequency domain plot
def plot_fft(x, fs, ax, label):
    N = len(x)
    X = fft(x)
    X_mag = np.abs(X[:N // 2])
    f = np.linspace(0, fs / 2, N // 2)
    ax.plot(f, X_mag, label=label)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')
    ax.legend()


fs = n / 10  # sampling frequency
plt.subplot(2, 2, 2)
plot_fft(window1, fs, plt.gca(), 'Window 1')
plot_fft(window2, fs, plt.gca(), 'Window 2')
plt.title('Frequency Domain')

# Print statistics
print(f"Window 1 - Mean: {np.mean(window1):.4f}, Variance: {np.var(window1):.4f}")
print(f"Window 2 - Mean: {np.mean(window2):.4f}, Variance: {np.var(window2):.4f}")

plt.tight_layout()
plt.show()
