from dataset.dataset_utils import get_glove_embeddings
from dataset.dataset_utils import get_glove_vocab_index_mapping
from torch.utils.data import DataLoader
from dataset.snli import SNLIDataset
from train_module.train_model import train_model
import os
import argparse

GLOVE_PATH = os.path.join("glove", "glove.840B.300d.txt")


def end_to_end_model_train(checkpoint_path, batch_size, encoder_name, lr,
                           weight_decay, num_epochs, hidden_size,
                           encoder_output_size,
                           lstm_bidirectional, num_workers):
    glove_vocab_index_mapping = get_glove_vocab_index_mapping(GLOVE_PATH)
    glove_embeddings = get_glove_embeddings(GLOVE_PATH)

    train_dataset = SNLIDataset("train", glove_vocab_index_mapping)
    validation_dataset = SNLIDataset("validation", glove_vocab_index_mapping)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    train_model(
        encoder_name=encoder_name,
        data_loaders=[train_loader, validation_loader],
        glove_embeddings=glove_embeddings,
        bidirectional=lstm_bidirectional,
        encoder_output_size=encoder_output_size,
        hidden_size=hidden_size,
        checkpoint_path=checkpoint_path,
        lr=lr,
        weight_decay=weight_decay,
        num_epochs=num_epochs,
    )


parser = argparse.ArgumentParser(
    description="Hyperparameters used for training")

parser.add_argument("--checkpoint_path", type=str, default=None,
                    help="Pre-trained model path if you want to resume training")

parser.add_argument("--batch_size", type=int, default=64,
                    help="Batch size used for data loading")

parser.add_argument("--encoder_name", type=str,
                    help="The encoder to use for sentence representation")

parser.add_argument("--lr", type=float, default=0.1,
                    help="Learning rate")

parser.add_argument("--weight_decay", type=float, default=0.99,
                    help="Learning rate weight decay")

parser.add_argument("--num_epochs", type=int, default=20,
                    help="The max number of epochs to train the model for")

parser.add_argument("--encoder_output_size", type=int,
                    help="Encoder output representation size")

parser.add_argument("--lstm_bidirectional", type=bool, default=False,
                    help="Use bidirectional LSTM")

parser.add_argument("--num_workers", type=int, default=6,
                    help="Number of workers to use for loading data")

parser.add_argument("--hidden_size", type=int, default=6,
                    help="LSTM hidden size")

args = parser.parse_args()

end_to_end_model_train(
    encoder_name=args.encoder_name,
    batch_size=args.batch_size,
    lstm_bidirectional=args.lstm_bidirectional,
    encoder_output_size=args.encoder_output_size,
    checkpoint_path=args.checkpoint_path,
    lr=args.lr,
    weight_decay=args.weight_decay,
    num_epochs=args.num_epochs,
    num_workers=args.num_workers,
    hidden_size=args.hidden_size
)

# re-train
# implement evaluate script (test snli + senteval)
# report test acc
# readme
# upload models + tensorboard runs
# notebook (be able to load model + do inference on 2 sentences)
# error analysis...
# add comments to code pls
# fix pep8