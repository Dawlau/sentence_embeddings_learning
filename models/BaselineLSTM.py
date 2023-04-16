import torch.nn as nn
import torch
from models.BaselineEncoder import BaselineEncoder


class BaselineLSTM(BaselineEncoder):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1,
                 bidirectional=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx)
        self.lstm_layer = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=bidirectional
        )

    def forward(self, x):
        embeddings = self.embeddings(x)
        hidden_states = self.lstm_layer(embeddings)[0]
        last_hidden_state = hidden_states[:, -1, :]

        return last_hidden_state
