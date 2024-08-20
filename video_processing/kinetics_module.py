import os
import pytorch_lightning
import pytorchvideo.data
import torch.utils.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip
)

class KineticsDataModule(pytorch_lightning.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Kinetics dataset.
    This module handles data loading and preprocessing for training and validation.
    """

    # Dataset configuration
    _DATA_PATH = <path_to_kinetics_data_dir>
    _CLIP_DURATION = 2  # Duration of sampled clip for each video
    _BATCH_SIZE = 8
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
        Create the Kinetics train data loader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the training dataset
        """
        train_transform = Compose([
            ApplyTransformToKey(
                key="video",
                transform=Compose([
                    UniformTemporalSubsample(8),
                    Lambda(lambda x: x / 255.0),
                    Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                    RandomShortSideScale(min_size=256, max_size=320),
                    RandomCrop(244),
                    RandomHorizontalFlip(p=0.5),
                ]),
            ),
        ])
        
        train_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "train.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("random", self._CLIP_DURATION),
            transform=train_transform
        )
        
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation data loader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the validation dataset
        """
        val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self._DATA_PATH, "val"),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            decode_audio=False,
        )
        
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )