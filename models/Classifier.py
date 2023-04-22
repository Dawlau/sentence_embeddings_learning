import torch.nn as nn


class Classifier(nn.Module):
    """
    A simple feedforward neural network for classification tasks.

    Args:
        input_size (int): The size of the encoder output.

    Methods:
        forward(x): Forward pass of the classifier.

    """
    def __init__(self, input_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        """
        Forward pass of the classifier.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes).

        """
        return self.layers(x)
