import torch.nn as nn
import torch


class BaselineEncoder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=1):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

    def forward(self, x):
        return torch.mean(self.embeddings(x), dim=1)

    def set_embeddings(self, embeddings, non_trainable=True):
        self.embeddings.load_state_dict({
            "weight": torch.tensor(embeddings)
        })

        if non_trainable:
            self.embeddings.weight.requires_grad = False
