import torch
import torch.fft

def compute_stft(signal, n_fft=1024, hop_length=512):
    return torch.fft.fft(signal, n=n_fft, dim=-1)


padded_signal = torch.randn(10, 100, 3)
# Example usage
stft_result = compute_stft(padded_signal)
print(stft_result.shape)

no_stft = torch.fft.fft(padded_signal, dim=-1)
print(no_stft.shape)