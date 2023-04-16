import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        return self.layers(x)
