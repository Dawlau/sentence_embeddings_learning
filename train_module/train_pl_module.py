import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from models.Classifier import Classifier
from models.utils import get_encoder


class SNLIModule(pl.LightningModule):
    def __init__(self, encoder_name, glove_embeddings, num_embeddings,
                 embedding_dim, bidirectional, encoder_output_size):
        super().__init__()
        self.save_hyperparameters()

        self.encoder = get_encoder(
            encoder_name, num_embeddings, embedding_dim, bidirectional)
        self.encoder.set_embeddings(glove_embeddings)
        self.classifier = Classifier(encoder_output_size)

        self.encoder.to(self.device)
        self.classifier.to(self.device)

        self.loss_module = nn.CrossEntropyLoss()

        self.validation_accuracies = [-1, -1]  # only keep the last 2 epochs

        self.validation_labels = torch.tensor([]).to(self.device)
        self.validation_preds = torch.tensor([]).to(self.device)

    def forward(self, premises, hypotheses):
        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)

        u = self.encoder(premises)
        v = self.encoder(hypotheses)

        encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

        return self.classifier(encoder_output)

    def configure_optimizers(self):
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        return [optimizer], []

    def training_step(self, batch, batch_idx):
        premises, hypotheses, labels = batch

        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)
        labels = labels.to(self.device)

        u = self.encoder(premises)
        v = self.encoder(hypotheses)

        encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

        preds = self.classifier(encoder_output)

        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard
        self.log('train_acc', acc)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        premises, hypotheses, labels = batch

        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)
        labels = labels.to(self.device)

        u = self.encoder(premises)
        v = self.encoder(hypotheses)

        encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

        preds = self.classifier(encoder_output)
        preds = preds.argmax(dim=-1)

        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.validation_labels = torch.cat([
            self.validation_labels.to(self.device),
            labels
        ])

        self.validation_preds = torch.cat([
            self.validation_preds.to(self.device),
            preds
        ])

        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        premises, hypotheses, labels = batch

        premises = premises.to(self.device)
        hypotheses = hypotheses.to(self.device)
        labels = labels.to(self.device)

        u = self.encoder(premises)
        v = self.encoder(hypotheses)

        encoder_output = torch.cat([u, v, torch.abs(u - v), u * v], dim=1)

        preds = self.classifier(encoder_output)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log('test_acc', acc)

    def on_train_epoch_end(self):
        optimizer = self.optimizers()

        optimizer.param_groups[0]["lr"] = 0.99 * optimizer.param_groups[0]["lr"]

    def on_validation_epoch_end(self):
        optimizer = self.optimizers()

        acc = (self.validation_labels == self.validation_preds) \
            .cpu().float().mean()

        self.validation_accuracies[0] = self.validation_accuracies[1]
        self.validation_accuracies[1] = acc

        if acc < self.validation_accuracies[0]:
            optimizer.param_groups[0]["lr"] /= 5

        self.validation_labels = torch.tensor([]).to(self.device)
        self.validation_preds = torch.tensor([]).to(self.device)

        if optimizer.param_groups[0]["lr"] < 1e-5:
            self.trainer.should_stop = True
