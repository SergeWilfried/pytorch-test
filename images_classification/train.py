import torch
from torchvision import transforms, datasets, models
import lightning as L


class LitSegmentation(L.LightningModule):
    """
    Lightning module for image segmentation using FCN-ResNet50.
    """
    def __init__(self):
        super().__init__()
        self.model = models.segmentation.fcn_resnet50(num_classes=21)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def training_step(self, batch):
        images, targets = batch
        outputs = self.model(images)['out']
        loss = self.loss_fn(outputs, targets.long().squeeze(1))
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    
class SegmentationData(L.LightningDataModule):
    """
    Lightning data module for VOC segmentation dataset.
    """
    def prepare_data(self):
        datasets.VOCSegmentation(root="data", download=True)
    
    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256, 256)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        target_transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((256, 256))])
        train_dataset = datasets.VOCSegmentation(root="data", transform=transform, target_transform=target_transform)
        return torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
    


if __name__ == "__main__":
    """
    Main execution:
    1. Initialize model and data modules
    2. Create a Lightning Trainer
    3. Start model training
    """
    model = LitSegmentation()
    data = SegmentationData()
    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, data)