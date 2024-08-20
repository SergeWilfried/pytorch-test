from video_processing.classification import VideoClassificationLightningModule
from video_processing.kinetics_module import KineticsDataModule
import pytorch_lightning

def train():
    """
    Main function to set up and run the video classification training process.

    This function initializes the classification module, data module, and trainer,
    then starts the model training.
    """
    # Initialize the video classification module
    classification_module = VideoClassificationLightningModule()

    # Initialize the Kinetics data module
    data_module = KineticsDataModule()

    # Set up the PyTorch Lightning trainer
    trainer = pytorch_lightning.Trainer()

    # Start the training process
    trainer.fit(classification_module, data_module)