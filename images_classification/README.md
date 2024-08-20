# Image Segmentation with FCN-ResNet50

This project implements semantic image segmentation using a Fully Convolutional Network (FCN) with ResNet50 backbone on the Pascal VOC dataset.

## Files

- `train.py`: Defines the training pipeline using PyTorch Lightning.
- `test.py`: Performs inference on a single image and visualizes the results.

## Setup

1. Install dependencies:
   ```
   pip install torch torchvision pytorch-lightning matplotlib pillow
   ```

2. Download the Pascal VOC dataset (automatic when running `train.py`).

## Usage

### Training

Run the training script:

```
python train.py
```

This will:
- Download the Pascal VOC dataset
- Train the FCN-ResNet50 model for 10 epochs
- Save checkpoints in the `lightning_logs` directory

### Inference

To segment an image, run the testing script:

```
python test.py
```

This will:
- Load the pre-trained model
- Load and preprocess the input image
- Perform inference to get the segmentation prediction
- Visualize the original image and segmentation result side by side
- Save the visualization as "segmentation.png"
