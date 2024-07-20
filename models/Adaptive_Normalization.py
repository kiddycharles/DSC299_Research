import torch
import torch.nn as nn


class AdaptiveNormalizationModule(nn.Module):
    def __init__(self, revin, revin_freq):
        super(AdaptiveNormalizationModule, self).__init__()
        self.revin = revin
        self.revin_freq = revin_freq

        # The combination layer needs to match the feature dimension
        self.fc = nn.Linear(2, 1)  # This assumes feature concatenation. Adjust if needed.

    def forward(self, x_enc):
        # Apply time domain normalization
        x_enc_time = self.revin(x_enc, 'norm')  # Shape: [batch_size, sequence_len, num_features]

        # Apply frequency domain normalization
        x_enc_freq = self.revin_freq(x_enc, 'norm')  # Shape: [batch_size, sequence_len, num_features]

        # Combine features by concatenation and then apply a learnable weight
        # Concatenate along the feature dimension
        combined_features = torch.cat((x_enc_time, x_enc_freq),
                                      dim=-1)  # Shape: [batch_size, sequence_len, 2*num_features]

        # Compute attention weights
        # This requires reshaping combined_features to [batch_size*sequence_len, 2*num_features]
        combined_features_flat = combined_features.view(-1, combined_features.size(-1))
        weights = torch.sigmoid(self.fc(combined_features_flat))  # Shape: [batch_size*sequence_len, 1]

        # Reshape weights back to [batch_size, sequence_len, 1]
        weights = weights.view(x_enc.size(0), x_enc.size(1), -1)

        # Apply weights to the time and frequency domain normalizations
        x_enc_time_weighted = weights * x_enc_time
        x_enc_freq_weighted = (1 - weights) * x_enc_freq

        # Combine the weighted outputs
        weighted_combined = x_enc_time_weighted + x_enc_freq_weighted

        return weighted_combined
