from models.BaselineLSTM import BaselineLSTM
import torch


class MaxPoolingBiLSTM(BaselineLSTM):
    def forward(self, x):
        embeddings = self.embeddings(x)
        hidden_states = self.lstm_layer(embeddings)[0]
        sentence_representation, _ = torch.max(hidden_states, dim=1)

        return sentence_representation
