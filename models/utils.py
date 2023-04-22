from models.BaselineEncoder import BaselineEncoder
from models.BaselineLSTM import BaselineLSTM
from models.MaxPoolingBiLSTM import MaxPoolingBiLSTM


def get_encoder(encoder_name, num_embeddings,
                embedding_dim, hidden_size=2048, bidirectional=False):
    """
    Return an encoder model based on the encoder_name input.

    Args:
        encoder_name (str): Name of the encoder model to be retrieved.
        num_embeddings (int): Number of tokens in the embeddings vocabulary.
        embedding_dim (int): Dimension of the embeddings.
        hidden_size (int): Size of the LSTM hidden states.
        bidirectional (bool): Determines if the encoder is bidirectional.

    Returns:
        Encoder model based on the encoder_name input.
    """
    if encoder_name == "BaselineEncoder":
        return BaselineEncoder(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
    elif encoder_name == "BaselineLSTM":
        return BaselineLSTM(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            bidirectional=bidirectional,
            hidden_size=hidden_size
        )
    else:
        return MaxPoolingBiLSTM(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            bidirectional=bidirectional,
            hidden_size=hidden_size
        )
