import pytorch_lightning as pl

from models.model_factory import get_model_by_config
from models.optimizer_factory import get_optimizer_by_config


class LitSimpleClassificationModel(pl.LightningModule):
    def __init__(
            self,
            model_config,

    ):
        super().__init__()

        self.model = get_model_by_config(model_config["classifier"])
        self.optimizer_config = model_config["optimizer"]

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = get_optimizer_by_config(self.parameters(), self.optimizer_config)
        return optimizer
