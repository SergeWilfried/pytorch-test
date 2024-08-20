import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from PIL import Image
from train import LitSegmentation

# Replace with path to your trained checkpoint 'lightning_logs/version_x/checkpoints/...'
# If not spedified, will use pretrained weights from TorchVision
checkpoint_path = None

# Path to the input image to segment
input_image_path = "image.jpg"


# Define the colormap for Pascal VOC (21 classes including background)
VOC_COLORMAP = np.array(
    [
        [0, 0, 0],  # Background
        [128, 0, 0],  # Aeroplane
        [0, 128, 0],  # Bicycle
        [128, 128, 0],  # Bird
        [0, 0, 128],  # Boat
        [128, 0, 128],  # Bottle
        [0, 128, 128],  # Bus
        [128, 128, 128],  # Car
        [64, 0, 0],  # Cat
        [192, 0, 0],  # Chair
        [64, 128, 0],  # Cow
        [192, 128, 0],  # Dining table
        [64, 0, 128],  # Dog
        [192, 0, 128],  # Horse
        [64, 128, 128],  # Motorbike
        [192, 128, 128],  # Person
        [0, 64, 0],  # Potted plant
        [128, 64, 0],  # Sheep
        [0, 192, 0],  # Sofa
        [128, 192, 0],  # Train
        [0, 64, 128],  # TV/Monitor
    ]
)


def voc_colormap2label():
    """Create a colormap to label mapping for Pascal VOC."""
    colormap2label = np.zeros(256**3, dtype=np.uint8)
    for i, colormap in enumerate(VOC_COLORMAP):
        colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i
    return colormap2label


colormap2label = voc_colormap2label()


def label2image(pred):
    """Convert label to color image."""
    colormap = VOC_COLORMAP[pred]
    return colormap.astype(np.uint8)


# Load the model
if checkpoint_path is not None:
    model = LitSegmentation.load_from_checkpoint(checkpoint_path)
else:
    model = LitSegmentation()
    # Use the pretrained weights from TorchVision
    model.model = models.segmentation.fcn_resnet50(
        num_classes=21, weights=models.segmentation.FCN_ResNet50_Weights.DEFAULT
    )
model.eval()

# Load the input image
input_image = Image.open(input_image_path).convert("RGB")

# Preprocess the input image
preprocess = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

input_tensor = preprocess(input_image).unsqueeze(0)  # Add batch dimension

# Get the model prediction
with torch.no_grad():
    output = model.model(input_tensor)["out"]
pred = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# Convert label to color image
pred_colored = label2image(pred)

# Visualize the input image and the segmentation result
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Input image
axs[0].imshow(transforms.Resize((256, 256))(input_image))
axs[0].set_title("Input Image", color="white")
axs[0].axis("off")

# Segmentation result
axs[1].imshow(pred_colored)
axs[1].set_title("Segmentation Result", color="white")
axs[1].axis("off")

plt.show()
plt.gcf().set_facecolor("black")
plt.savefig("segmentation.png")

# ... imports and initial setup ...

# Define Pascal VOC colormap and helper functions
VOC_COLORMAP = np.array([...])

def voc_colormap2label():
    """Create a colormap to label mapping for Pascal VOC."""
    # ... implementation ...

def label2image(pred):
    """Convert label to color image."""
    # ... implementation ...

# Load the model
# ... model loading code ...

# Load and preprocess the input image
# ... image loading and preprocessing ...

# Get the model prediction
# ... prediction code ...

# Visualize the input image and the segmentation result
# ... visualization code ...

# Save the result
plt.savefig("segmentation.png")