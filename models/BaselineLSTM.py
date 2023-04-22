import torch.nn as nn
from models.BaselineEncoder import BaselineEncoder


class BaselineLSTM(BaselineEncoder):
    """
    Module that implements a baseline LSTM encoder for sentence embeddings.

    Args:
        num_embeddings (int): Number of embeddings in the input vocabulary.
        embedding_dim (int): Dimensionality of the input embeddings.
        hidden_size (int): Number of features in the hidden state of the LSTM.
        padding_idx (int, optional): Index of the padding token in the input
            vocabulary. Default is 1.
        bidirectional (bool, optional): If True, use a bidirectional LSTM.
            Default is False.

    Methods:
        forward(x): Defines the forward pass.
    """
    def __init__(self, num_embeddings, embedding_dim, hidden_size,
                 padding_idx=1, bidirectional=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )

    def forward(self, x):
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): A tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, hidden_size).
        """
        embeddings = self.embeddings(x)
        hidden_states = self.lstm_layer(embeddings)[0]
        last_hidden_state = hidden_states[:, -1, :]

        return last_hidden_state
