import pytorch_lightning as pl
import torch.nn

from models.model_factory import get_model_by_config
from models.optimizer_factory import get_optimizer_by_config


class LitBasicClassificationModel(pl.LightningModule):
    def __init__(
            self,
            model_config,

    ):
        super().__init__()

        self.model = get_model_by_config(model_config["classifier"])
        self.optimizer_config = model_config["optimizer"]

        self.loss = torch.nn.CrossEntropyLoss()

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = get_optimizer_by_config(self.parameters(), self.optimizer_config)
        return optimizer

    def model_step(self, images, target_labels: torch.Tensor):
        resulted_classes = self.model(images)
        assert resulted_classes.ndim == 2  # Sanity check

        loss = self.loss(resulted_classes, target_labels)

        predicted_labels = target_labels.argmax(-1)
        accuracy = (target_labels == predicted_labels).mean()

        return loss, accuracy

    def training_step(self, batch, *args, **kwargs):
        train_loss, train_accuracy = self.model_step(*batch)

        self.log("train_loss", train_loss, prog_bar=True)
        self.log("train_accuracy", train_accuracy, prog_bar=True)

    def validation_step(self, batch, *args, **kwargs):
        val_loss, val_accuracy = self.model_step(*batch)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_accuracy", val_accuracy, prog_bar=True)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.model(batch)
