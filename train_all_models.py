from dataset.dataset_utils import get_glove_embeddings
from dataset.dataset_utils import get_glove_vocab_index_mapping
from torch.utils.data import DataLoader
from dataset.snli import SNLIDataset
from train_module.train_model import train_model
import os

GLOVE_PATH = os.path.join("glove", "glove.840B.300d.txt")
BATCH_SIZE = 64
NUM_WORKERS = 6
MODELS_CONFIG = {
    "BaselineEncoder": {
        "output_size": 1200,
        "bidirectional": False
    },
    "BaselineLSTM": {
        "output_size": 1200,
        "bidirectional": False
    },
    "BidirectionalLSTM": {
        "output_size": 2400,
        "bidirectional": True
    },
    "MaxPoolingBiLSTM": {
        "output_size": 2400,
        "bidirectional": True
    }
}


def train_all_models():
    glove_vocab_index_mapping = get_glove_vocab_index_mapping(GLOVE_PATH)
    glove_embeddings = get_glove_embeddings(GLOVE_PATH)

    train_dataset = SNLIDataset("train", glove_vocab_index_mapping)
    validation_dataset = SNLIDataset("validation", glove_vocab_index_mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    for model_name, config in MODELS_CONFIG.items():
        output_size, bidirectional = config.values()

        train_model(
            model_name,
            data_loaders=[train_loader, validation_loader],
            glove_embeddings=glove_embeddings,
            num_embeddings=glove_embeddings.shape[0],
            embedding_dim=glove_embeddings.shape[1],
            bidirectional=bidirectional,
            encoder_output_size=output_size
        )

if __name__ == "__main__":
    train_all_models()
