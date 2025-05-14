# Machine Learning Assignment 2 - Classifying disease task

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
   Place your images in `data/train_images/` and ensure metadata like `meta_train.csv` is available.

2. **Train a model**  
   Use the training scripts in `scripts/train/`:
   - `train_classify_diseases.py`: Train on full images
   - `train_masked_model.py`: Train on lesion-masked images
   - `new_data_train.py`: Train or fine-tune on new dataset

3. **Evaluate or fine-tune**  
   Use the `test/` folder:
   - `testing.py`: Run model on test set and generate metrics
   - `new_data_tune.py`: Fine-tune pre-trained model on a new dataset

4. **Visualize training**  
   Training history is saved in `checkpoints/visualizations/` as `.json` files and can be plotted using the tools in `utils/`.

5. **Model checkpoints**  
   Saved `.pt` model files can be found in the `checkpoints/` folder.

## How to Run
pip install -r requirements.txt
python -m main --mode <model to train>

The arguments after --mode can be --raw, --masked or --new_data, to run the scripts to train the  models accordingly. 