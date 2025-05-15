This folder contains training scripts used to build classification models for rice disease detection. Each script is designed for a specific training configuration, including both the main supervised training and an experimental setup using masked lesion images for comparative analysis.



### train_classify_diseases.py
- This is the main training script for the rice disease classification model.
- Trains a CBAM-ResNet34 model using augmented images.
- Hyperparameters are tuned directly in this script in the development phase of the model
### train_masked_model.py
- This script is where the same model architecture is used to train on the masked dataset for experimental purpose, after literature review. 
### new_data_train.py
- This is where I fine-tuned the 2 models on the completely new dataset I found online, which features a different concept of crops data: closed-up view instead of wide on-field view. To start fine-tuning backbone model and dataset used, just fill in the arguments at the end of the code.  
### train_independent.py
- This is the attempt of the recreation of the pre-trained model proposed from the research paper, who is trying to solve the same classifying disease as me. The model is recreated for independent comparison and evaluation to my own developed model only. 
## Notes
- All scripts use datasets under the `data/` directory.
- Model checkpoints and training history will be saved to the `checkpoints/` directory.
- The scripts uses defined function from other folders like src, test and utils
- I tuned the hyper-parameters directly in these script, simply by filling in the hyperparameters in the arguments. The current code is the setting I used to produced the final model in every categories. 
## Running the Script
python -m scripts.train.train_classify_diseases
python -m scripts.train.train_masked_model
python -m scripts.train.new_data_train
python -m scripts.train.train_independent