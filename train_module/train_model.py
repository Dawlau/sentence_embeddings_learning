import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from train_module.train_pl_module import SNLIModule
import os
import torch


CHECKPOINT_PATH = os.path.join("saved_models")
NUM_EPOCHS = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

def train_model(model_name, data_loaders, glove_embeddings, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, model_name),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices="auto",
        max_epochs=NUM_EPOCHS,
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="val_acc"
            ),
        ],
        enable_progress_bar=True
    )

    train_loader, validation_loader = data_loaders

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = SNLIModule.load_from_checkpoint(pretrained_filename)
    else:
        pl.seed_everything(42)
        model = SNLIModule(
            encoder_name=model_name,
            glove_embeddings=glove_embeddings,
            **kwargs
        )
        trainer.fit(model, train_loader, validation_loader)