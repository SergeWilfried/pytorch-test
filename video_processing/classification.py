import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning
import pytorchvideo.models.resnet

def make_kinetics_resnet():
    return pytorchvideo.models.resnet.create_resnet(
        input_channel=3,  # RGB input from Kinetics
        model_depth=50,  # Using a 50 layer network
        model_num_class=400,  # Kinetics has 400 classes
        norm=nn.BatchNorm3d,
        activation=nn.ReLU,
    )

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    """
    A PyTorch Lightning module for video classification tasks.
    This module encapsulates a video classification model based on a Kinetics ResNet architecture.
    """

    def __init__(self):
        """
        Initialize the module with a Kinetics ResNet model.
        """
        super().__init__()
        self.model = make_kinetics_resnet()

    def forward(self, x):
        """
        Define the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Output of the model
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step.

        Args:
            batch (dict): A dictionary containing 'video' and 'label' tensors
            batch_idx (int): The index of the current batch

        Returns:
            torch.Tensor: The computed loss
        """
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.cross_entropy(y_hat, batch["label"])

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Perform a single validation step.

        Args:
            batch (dict): A dictionary containing 'video' and 'label' tensors
            batch_idx (int): The index of the current batch

        Returns:
            torch.Tensor: The computed loss
        """
        y_hat = self.model(batch["video"])
        loss = F.cross_entropy(y_hat, batch["label"])
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        """
        Set up the optimizer for training.

        Returns:
            torch.optim.Optimizer: The configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=1e-1)
    
