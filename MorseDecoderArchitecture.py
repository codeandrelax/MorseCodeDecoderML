
"""
MorseDecoderArchitecture.py

This script defines a BiLSTM-based neural network architecture for Morse code decoding.
The architecture consists of several dense layers followed by a bidirectional LSTM.
The output is processed using a softmax function to provide log-probabilities for Connectionist Temporal Classification (CTC).

Author: Damjan Prerad
Date: 2025-05-13

Example usage:
# To create the model
model = BiLSTMRnn(num_tags=5, spectrogram_size=64)

# To get model output
output = model(input_tensor)
"""

import torch.nn as nn
import torch.nn.functional as F

class BiLSTMRnn(nn.Module):
    """
    A BiLSTM-based architecture for decoding Morse code from spectrogram input.

    Parameters
    ----------
    num_tags : int
        The number of output tags (labels), including a blank token for CTC.
    spectrogram_size : int
        The size of the spectrogram input features.
    """
    def __init__(self, num_tags, spectrogram_size):
        super(BiLSTMRnn, self).__init__()

        num_tags += 1  # 0: blank for CTC
        hidden_dim = 256
        lstm_dim = 256

        self.dense1 = nn.Linear(spectrogram_size, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(0.1)

        self.dense3 = nn.Linear(hidden_dim, lstm_dim)

        # Bidirectional LSTM
        self.lstm = nn.LSTM(input_size=lstm_dim,
                            hidden_size=lstm_dim,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.1)

        self.output = nn.Linear(lstm_dim * 2, num_tags)

    def forward(self, x):
        """
        Forward pass for the BiLSTM model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape [batch, seq_len, spectrogram_size].
            It is expected to be a batch of spectrograms.

        Returns
        -------
        torch.Tensor
            The output tensor with shape [batch, seq_len, num_tags], containing 
            the log-probabilities for each class at each time step.
        """
        # x: [batch, seq_len, spectrogram_size]
        B, T, feat_dim = x.size()
        x = x.view(-1, feat_dim)               # (B*T, feat_dim)
        x = F.relu(self.ln1(self.dense1(x)))
        x = self.dropout1(x)
        x = F.relu(self.ln2(self.dense2(x)))
        x = self.dropout2(x)
        x = F.relu(self.dense3(x))
        x = x.view(B, T, -1)            # (B, T, lstm_dim)

        x, _ = self.lstm(x)             # (B, T, 2*lstm_dim)
        x = self.output(x)             # (B, T, num_tags)
        x = F.log_softmax(x, dim=2)    # log-probabilities for CTC
        return x

    def count_parameters(self):
        """
        Counts the number of trainable parameters in the model.

        Returns
        -------
        int
            The number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    