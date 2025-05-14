This module contains the core source code for training and evaluating classification models used in the project. All the code defined in this folder are used in the scripts to train and develop models. 

## Structure Overview

### models/
Contains all model architectures used for experimentation and comparison.
- `simple_cnn.py`  
  A basic convolutional neural network architecture used for the baseline model.

- `resnet_model.py`  
  Implements ResNet-based with no pre-trained weights classifiers. I used ResNet18 for original development and tuning.

- `cbam.py`  
  Defines the CBAM (Convolutional Block Attention Module) model architecture that enhances ResNet with spatial and channel-wise attention.

### trainer.py
Implements the training loop with support for CutMix, MixUp, early stopping, and learning rate scheduling. Tracks training/validation metrics and saves both model weights and training history as JSON.

- `train_model(...)`: Core training function.
- Supports: early stopping (from utils), CutMix, MixUp, scheduler.
- Outputs: model `.pt` file + `history_*.json`.
- 
### eval.py
Provides evaluation and visualization tools for trained models.

- `evaluate_model(...)`: Runs evaluation, prints metrics, plots confusion matrix.
- `plot_confusion_matrix(...)`: Visualizes predictions.
- `plot_training_history(...)`: Plots loss and accuracy curves from saved logs.