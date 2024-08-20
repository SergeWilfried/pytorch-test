# Computer Vision with PyTorch Lightning

This project implements both image segmentation and video classification using PyTorch Lightning.

## Components

1. Image Segmentation
   - Uses FCN-ResNet50 on Pascal VOC dataset
   - Files: `images_classification/train.py`, `images_classification/test.py`

2. Video Classification
   - Uses Kinetics ResNet on Kinetics dataset
   - Files: `video_processing/kinetics_module.py`, `video_processing/classification.py`, `video_processing/run.py`

## Setup

1. Install dependencies:
   ```
   pip install torch torchvision pytorch-lightning matplotlib pillow pytorchvideo
   ```

2. Datasets:
   - Pascal VOC: Downloaded automatically when running image segmentation training
   - Kinetics: Update `_DATA_PATH` in `kinetics_module.py`

## Usage

### Image Segmentation

1. Train: `python images_classification/train.py`
2. Test: `python images_classification/test.py`

### Video Classification

Run the training script:

```
python video_processing/run.py
```

This script initializes the classification module, data module, and trainer, then starts the model training.

## Customization

- Adjust hyperparameters in respective modules
- Modify model architectures as needed

## License

MIT License
