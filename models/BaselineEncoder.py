import torch.nn as nn
import torch


class BaselineEncoder(nn.Module):
    """
    A simple baseline encoder that averages the embeddings of input sequences.

    Args:
        num_embeddings (int): Size of the vocabulary.
        embedding_dim (int): Dimensionality of the embeddings.
        padding_idx (int, optional): The index of the padding token in
            the vocabulary. Defaults to 1.

    Methods:
        forward(x): Defines the forward pass of the model.
        set_embeddings(embeddings, non_trainable=True): Initializes the
            embedding layer with pre-trained embeddings.

    """
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, x):
        """
        Forward pass of the BaselineEncoder.

        Args:
            x (Tensor): Batch of input sequences. Shape: (batch_size, seq_len).

        Returns:
            Tensor: Output of the encoder. Shape: (batch_size, embedding_dim).

        """
        return torch.mean(self.embeddings(x), dim=1)

    def set_embeddings(self, embeddings, non_trainable=True):
        """
        Initializes the embedding layer with pre-trained embeddings.

        Args:
            embeddings (ndarray): Pre-trained embeddings.
            non_trainable (bool, optional): If True, the embeddings will be
                non-trainable. Defaults to True.

        """
        self.embeddings.load_state_dict({
            "weight": torch.tensor(embeddings)
        })

        if non_trainable:
            self.embeddings.weight.requires_grad = False
