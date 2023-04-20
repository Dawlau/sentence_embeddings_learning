from models.BaselineEncoder import BaselineEncoder
from models.BaselineLSTM import BaselineLSTM
from models.MaxPoolingBiLSTM import MaxPoolingBiLSTM


def get_encoder(encoder_name, num_embeddings,
                embedding_dim, hidden_size=2048, bidirectional=False):
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
