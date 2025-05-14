# utils/

This folder contains utility modules used throughout the training and evaluation pipeline for rice disease classification. These utilities support data loading, model inspection, custom loss functions, training control mechanisms, and result visualization.

## File Overview

### `datamodule.py`
Contains utility functions and classes for loading and preparing datasets. Handles:
- Dataset splitting (train/val/test)
- Image augmentations
- Weighted sampling for class imbalance
- DataLoader setup with PyTorch's `ImageFolder` and custom samplers

### `early_stopping.py`
Implements early stopping to prevent overfitting. Monitors validation loss and stops training if performance stops improving after a given number of epochs.

### `focal_loss.py`
Provides a custom implementation of Focal Loss, which is useful for handling class imbalance by focusing the loss on harder-to-classify examples.

### `gradcam.py`
Generates Grad-CAM visualizations to help interpret model decisions. Shows where the model is "looking" when making predictions, which is especially useful for debugging and explainability.

### `inspectors.py`
Used to plot results and visualization of evaluation after training models. Can also be used to visualize training history of the models variations.  

## Usage

These utilities are meant to be imported in your main training and evaluation scripts