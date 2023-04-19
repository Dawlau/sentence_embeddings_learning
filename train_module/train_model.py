from models.utils import get_encoder
from models.Classifier import Classifier
from train_module.train_utils import train_step
from train_module.train_utils import validation_step
from torch.utils.tensorboard import SummaryWriter
import os
import torch


CHECKPOINT_PATH = os.path.join("saved_models")
NUM_EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(model_name, data_loaders, glove_embeddings, **kwargs):
    if not os.path.isdir(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    encoder = get_encoder(
        encoder_name=model_name,
        num_embeddings=glove_embeddings.shape[0],
        embedding_dim=glove_embeddings.shape[1],
        bidirectional=kwargs["bidirectional"]
    )

    classifier = Classifier(kwargs["encoder_output_size"])

    encoder.to(device)
    classifier.to(device)

    optimizer = torch.optim.SGD(
        list(classifier.parameters()),
        lr=0.1
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter()
    validation_accuracies = [-1, -1]

    best_accuracy = -1

    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch}")

        train_loss, train_acc = train_step(
            encoder=encoder,
            classifier=classifier,
            dataloader=data_loaders[0],
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device
        )

        optimizer.param_groups[0]["lr"] = \
            0.99 * optimizer.param_groups[0]["lr"]

        validation_loss, validation_acc = validation_step(
            encoder=encoder,
            classifier=classifier,
            dataloader=data_loaders[1],
            loss_fn=loss_fn,
            device=device
        )

        if best_accuracy < validation_acc:
            torch.save(
                encoder, os.path.join(CHECKPOINT_PATH, model_name) + ".pt")
            best_accuracy = validation_acc

        validation_accuracies[0] = validation_accuracies[1]
        validation_accuracies[1] = validation_acc

        if validation_acc < validation_accuracies[0]:
            optimizer.param_groups[0]["lr"] /= 5

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', validation_acc, epoch)

        print(f"Finished epoch {epoch}")

        if optimizer.param_groups[0]["lr"] < 1e-5:
            break