import os.path
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers

import torch.utils.data

import augmentations
from datasets.dog_breed_identification_dataset import DogBreedIdentificationDataset


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--lightning_folder_path", required=True)
    parser.add_argument("--num_workers", required=False, default=0)
    parser.add_argument("--batch_size", required=False, default=32)
    parsed_args = parser.parse_args()

    common_dataset = DogBreedIdentificationDataset(
        parsed_args.dataset_path,
        image_transform=augmentations.get_augmentations()
    )

    train_dataset = torch.utils.data.Subset(common_dataset, range(0, len(common_dataset) - 1000))
    val_dataset = torch.utils.data.Subset(common_dataset, range(len(common_dataset) - 1000, len(common_dataset)))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, parsed_args.batch_size,
        shuffle=True, num_workers=parsed_args.num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, parsed_args.batch_size * 2,
        shuffle=False, num_workers=parsed_args.num_workers,
        pin_memory=True
    )

    logger = pl.loggers.TensorBoardLogger(parsed_args.lightning_folder_dir)
    checkpointer = pl.callbacks.ModelCheckpoint(os.path.join(parsed_args.lightning_folder_path, "checkpoints"))

    trainer = pl.Trainer(
        accelerator="cuda",
        logger=logger,
        callbacks=[checkpointer],
        log_every_n_steps=2,
        gradient_clip_val=1
    )





if __name__ == '__main__':
    main()
