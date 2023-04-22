from models.BaselineLSTM import BaselineLSTM
import torch


class MaxPoolingBiLSTM(BaselineLSTM):
    """
    A model representing a BiLSTM model with max-pooling for sentence
    representation.

    Args:
        num_embeddings (int): The size of the vocabulary.
        embedding_dim (int): The dimensionality of the word embeddings.
        hidden_size (int): The number of units in the LSTM hidden layer.
        padding_idx (int, optional): The index used for padding token.
            Defaults to 1.
        bidirectional (bool, optional): Whether to use a bidirectional LSTM.
            Defaults to False.
    """
    def forward(self, x):
        embeddings = self.embeddings(x)
        hidden_states = self.lstm_layer(embeddings)[0]
        sentence_representation, _ = torch.max(hidden_states, dim=1)

        return sentence_representation
