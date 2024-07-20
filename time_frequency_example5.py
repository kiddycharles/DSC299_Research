import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft

fs = 1000
t = np.linspace(0, 1, fs, endpoint=False)

# Time Domain Shift + Frequency Domain Shift
signal1 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.5, len(t))
signal1 = signal1 + np.linspace(0, 1, len(t))  # Mean shift

# Time Domain Shift + Frequency Domain No Shift
signal2 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t) + np.random.normal(0, 0.5, len(t))
signal2 = (signal2 - np.mean(signal2)) / np.std(signal2)  # Normalize

# Time Domain No Shift + Frequency Domain Shift
signal3 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 300 * t)
signal3 += np.random.normal(0, 0.5, len(t))

# Time Domain No Shift + Frequency Domain No Shift
signal4 = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 150 * t)

signals = [signal1, signal2, signal3, signal4]
titles = [
    'Time Domain Shift + Frequency Domain Shift',
    'Time Domain Shift + Frequency Domain No Shift',
    'Time Domain No Shift + Frequency Domain Shift',
    'Time Domain No Shift + Frequency Domain No Shift'
]

plt.figure(figsize=(14, 12))

for i, signal in enumerate(signals):
    fft_signal = fft(signal)
    frequencies = np.fft.fftfreq(len(t), 1 / fs)

    plt.subplot(4, 2, 2 * i + 1)
    plt.plot(t, signal)
    plt.title(f'{titles[i]} (Time Domain)')

    plt.subplot(4, 2, 2 * i + 2)
    plt.plot(frequencies, np.abs(fft_signal))
    plt.title(f'{titles[i]} (Frequency Domain)')
    plt.xlim(0, 200)

plt.tight_layout()
plt.show()
