import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second

# Create two signals
# Signal 1: A sine wave with a frequency of 5 Hz
signal1 = np.sin(2 * np.pi * 5 * t)

# Signal 2: A sum of two sine waves (5 Hz and 50 Hz)
signal2 = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 50 * t)

# Compute the Fourier Transform
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)

# Frequency vector
freq = np.fft.fftfreq(len(t), 1/fs)

# Plot time domain signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal1, label='Signal 1')
plt.title('Signal 1 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 2)
plt.plot(t, signal2, label='Signal 2', color='orange')
plt.title('Signal 2 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(freq, np.abs(fft_signal1), label='Signal 1')
plt.title('Signal 1 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 100)

plt.subplot(2, 2, 4)
plt.plot(freq, np.abs(fft_signal2), label='Signal 2', color='orange')
plt.title('Signal 2 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 100)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second

# Create two signals
# Signal 1: A sine wave with a frequency of 5 Hz
signal1 = np.sin(2 * np.pi * 5 * t)

# Signal 2: A sine wave with the same frequency but with a phase shift of 90 degrees (π/2 radians)
signal2 = np.sin(2 * np.pi * 5 * t + np.pi/2)

# Compute the Fourier Transform
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)

# Frequency vector
freq = np.fft.fftfreq(len(t), 1/fs)

# Plot time domain signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal1, label='Signal 1')
plt.title('Signal 1 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 2)
plt.plot(t, signal2, label='Signal 2', color='orange')
plt.title('Signal 2 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(freq, np.abs(fft_signal1), label='Signal 1')
plt.title('Signal 1 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 10)

plt.subplot(2, 2, 4)
plt.plot(freq, np.abs(fft_signal2), label='Signal 2', color='orange')
plt.title('Signal 2 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 10)

plt.tight_layout()
plt.show()


import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 1000  # Sampling frequency (Hz)
t = np.arange(0, 1, 1/fs)  # Time vector from 0 to 1 second

# Create two signals
# Signal 1: A sinusoid with a frequency of 5 Hz
signal1 = np.sin(2 * np.pi * 5 * t)

# Signal 2: A sinusoid with a frequency of 10 Hz, but the time window is the same
signal2 = np.sin(2 * np.pi * 10 * t)

# Compute the Fourier Transform
fft_signal1 = np.fft.fft(signal1)
fft_signal2 = np.fft.fft(signal2)

# Frequency vector
freq = np.fft.fftfreq(len(t), 1/fs)

# Plot time domain signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, signal1, label='Signal 1')
plt.title('Signal 1 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(2, 2, 2)
plt.plot(t, signal2, label='Signal 2', color='orange')
plt.title('Signal 2 (Time Domain)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# Plot frequency domain signals
plt.subplot(2, 2, 3)
plt.plot(freq, np.abs(fft_signal1), label='Signal 1')
plt.title('Signal 1 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 20)

plt.subplot(2, 2, 4)
plt.plot(freq, np.abs(fft_signal2), label='Signal 2', color='orange')
plt.title('Signal 2 (Frequency Domain)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.xlim(0, 20)

plt.tight_layout()
plt.show()

