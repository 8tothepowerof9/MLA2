# test/

This folder contains testing and tuning routines for both segmentation and classification models. I also tested different approach in this folder. This is where the experiments and iterative works happens, as I develop the models by trying difference approaches and combinations suggested by literature reviews. 

## File Descriptions

### `test_dataloader.py`
Tests the DataLoader setup and plot out a few image instance to make sure everything is read correctly. 

### `test_gpu.py`
Quick script to verify if GPU is available and properly utilized by PyTorch to avoid silent fallbacks to CPU.

### `test_rebalance.py`
Here is where I tried different balancing and augmenting methods for the dataloader. The current setting in this file is how I currently transform and preprocess my dataset. 

### `testing.py`
Runs simple test on data distributions, preprocessing results, ... I currently use this file to test the over/under and weights sampling methods. 

### `new_data_tune.py`
Handles fine-tuning of a pre-trained model on a new dataset (e.g., cross-dataset adaptation). Used to adapt the wide-field-trained model to the close-up dataset.

### `k_means.py`
Implements or tests a K-Means-based approach for masking the lesion marks on leaves. This is proposed on the referenced papers, however it did not work in my case since our dataset are in wider view, in contrast of the close up view in the paper's dataset. 

### `unet.py`
Testing and implementation of the U-net model used to generate masks of lesion leaves on the original dataset. 

## Typical Usage
python -m test.test_gpu
python -m test.test_dataloader
