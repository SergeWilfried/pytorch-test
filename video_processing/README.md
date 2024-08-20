# Video Classification with PyTorch Lightning

This project implements a video classification model using PyTorch Lightning, focusing on the Kinetics dataset.

## Overview

The project consists of two main components:

1. `KineticsDataModule`: Handles data loading and preprocessing for the Kinetics dataset.
2. `VideoClassificationLightningModule`: Implements the video classification model and training logic.

## Requirements

- PyTorch
- PyTorch Lightning
- PyTorchVideo
- torchvision

## File Structure

- `video_processing/`
  - `kinetics_module.py`: Contains the `KineticsDataModule` class
  - `classification.py`: Contains the `VideoClassificationLightningModule` class

## KineticsDataModule

This module is responsible for loading and preprocessing the Kinetics dataset. It provides data loaders for both training and validation sets.

Key features:
- Configurable clip duration, batch size, and number of workers
- Custom data transformations for training data
- Uniform temporal subsampling for consistent input sizes

## VideoClassificationLightningModule

This module encapsulates the video classification model based on a Kinetics ResNet architecture. It handles the training, validation, and optimization processes using PyTorch Lightning.

Key features:
- Implements forward pass, training step, and validation step
- Uses cross-entropy loss for classification
- Configures Adam optimizer for training

## Usage

To train the model:

1. Ensure the Kinetics dataset is available and update the `_DATA_PATH` in `kinetics_module.py`.
2. Create a script that initializes both modules and uses a PyTorch Lightning Trainer:

```python
from video_processing.classification import VideoClassificationLightningModule
from video_processing.kinetics_module import KineticsDataModule
import pytorch_lightning

def train():
    classification_module = VideoClassificationLightningModule()
    data_module = KineticsDataModule()
    trainer = pytorch_lightning.Trainer()
    trainer.fit(classification_module, data_module)

if __name__ == "__main__":
    train()
```

3. Run the script to start training.

## Customization

- Adjust the `_CLIP_DURATION`, `_BATCH_SIZE`, and `_NUM_WORKERS` in `KineticsDataModule` as needed.
- Modify the model architecture in `VideoClassificationLightningModule` by changing the `make_kinetics_resnet()` function.
- Experiment with different optimizers or learning rates in the `configure_optimizers()` method.

## License

MIT License
