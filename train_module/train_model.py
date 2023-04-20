from models.utils import get_encoder
from models.Classifier import Classifier
from train_module.train_utils import train_step
from train_module.train_utils import validation_step
from torch.utils.tensorboard import SummaryWriter
import os
import torch


SAVE_MODEL_PATH = os.path.join("saved_models")
device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def train_model(encoder_name, data_loaders, glove_embeddings,
                bidirectional, encoder_output_size, hidden_size,
                checkpoint_path, lr, weight_decay, num_epochs):

    if not os.path.isdir(SAVE_MODEL_PATH):
        os.mkdir(SAVE_MODEL_PATH)

    if checkpoint_path is not None:
        encoder = torch.load(
            os.path.join(checkpoint_path, encoder_name + ".pt")
        )

        classifier = torch.load(
            os.path.join(checkpoint_path, f"{encoder_name}_classifier" + ".pt")
        )
    else:
        encoder = get_encoder(
            encoder_name=encoder_name,
            num_embeddings=glove_embeddings.shape[0],
            embedding_dim=glove_embeddings.shape[1],
            hidden_size=hidden_size,
            bidirectional=bidirectional
        )

        classifier = Classifier(encoder_output_size)

    encoder.set_embeddings(glove_embeddings)

    encoder.to(device)
    classifier.to(device)

    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=lr
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    writer = SummaryWriter()

    last_validation_accuracy = -1
    best_accuracy = -1

    for epoch in range(num_epochs):
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
            weight_decay * optimizer.param_groups[0]["lr"]

        validation_loss, validation_acc = validation_step(
            encoder=encoder,
            classifier=classifier,
            dataloader=data_loaders[1],
            loss_fn=loss_fn,
            device=device
        )

        if best_accuracy < validation_acc:
            torch.save(
                encoder, os.path.join(SAVE_MODEL_PATH, encoder_name) + ".pt")
            torch.save(
                classifier, os.path.join(SAVE_MODEL_PATH, f"{encoder_name}_classifier") + ".pt")
            best_accuracy = validation_acc

        if validation_acc < last_validation_accuracy:
            optimizer.param_groups[0]["lr"] /= 5

        last_validation_accuracy = validation_acc

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/validation', validation_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/validation', validation_acc, epoch)

        print(f"Finished epoch {epoch}")

        if optimizer.param_groups[0]["lr"] < 1e-5:
            print("Learning rate is below 1e-5. Stoping...")
            break
