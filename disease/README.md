# Machine Learning Assignment 2 - Classifying disease task
## Make sure you cd to the disease/ directory before any running any command. 
## Before running any script, please put the dataset "train_images" and "test_images" inside disease/data folder. 
This repository contains a complete machine learning pipeline for the task classifying rice plant diseases from images, including literature reviews for lesion masking, transfer learning, and model interpretability. Here, I will be training a model directly on the provided dataset, and I will train another on the experimental dataset, which is the result from the literature review approach. Thus, I will call the models trained on normal data 'original models' and the other on masked data 'experimental models'.

## Project Structure

- `data/`  
  Stores datasets and metadata such as training images, experiment images and metadata.

- `scripts/`  
  Contains training scripts for both masked and unmasked datasets. Used to launch experiments and train models from scratch or on new data. As these main model are very bulky, I tuned them manually and directly through these scripts. 

- `src/`  
  Core source code for the classification model. Includes model definition, training/evaluation logic, and configuration setup.

- `test/`  
  Diagnostic scripts to test individual components like dataloaders, GPU availability, rebalancing strategies, and full model evaluation. Also includes scripts for fine-tuning on new datasets. This is where experimental models like unet and hyper-parameter tuning for the experiment models and different datasets outside of the scope happen. 

- `utils/`  
  Utility modules for common tasks like early stopping, GradCAM visualizations, focal loss, dataset inspection, and dataloading.

---

## Workflow Overview

1. **Prepare the dataset**  
   Takes images in `data/train_images/` and ensure metadata like `meta_train.csv` is available.
   The data will be processed by files under utils and test folder.

3. **Train and evaluate a model**  
   Use the training scripts in `scripts/train/`:
   - `train_classify_diseases.py`: Train on full images
   - `train_masked_model.py`: Train on lesion-masked images
   - `new_data_train.py`: Train or fine-tune on new dataset

5. **Visualize training**  
   Training history is saved in `checkpoints/visualizations/` as `.json` files and can be plotted using the tools in `utils/`.

6. **Model checkpoints**  
   Saved `.pt` model files can be found in the `checkpoints/` folder.

## How to Run
Make sure you are at the disease/ directory before any running any command. 
pip install -r requirements.txt
python -m main --mode <model to train>

or any file can be run by syntax: 
python -m <folder>.<filename> (no .py)

The arguments after --mode can be --raw, --masked or --new_data, to run the scripts to train the  models accordingly. 
