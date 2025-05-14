# MachineLearning-A2

## Setup for the predicting age task

First, put the data in to the `data` folder inside the `age` folder:
 
The correct path should be: `age\data\test_images` and `age\data\train_images`

Try to install all the requiremnt:
`pip install -r age\requirements.txt`

## Example to run the code

Fit Age Prediction Model:

`python age/main.py --task train --config age/config/config.yaml`

Run Test Age Model on unlabeled dataset:

`python age/main.py --task eval --config age/config/config.yaml`

Run Test Age Model for a random picture:

`python age/src/cbam34.py --image age\data\train_images\hispa\101315.jpg --model age\checkpoints\cbam34_model\best_model.pt`

## Configure the code

In order to change to parameter, go to the file:

`age\config\config.yaml`

The current setup is used for training the final model on the non weighted sampling dataset and plotting the training history. All results are stored in the checkpoints directory.

You can turn the `grad-camp` to True and change `grad_camp_sample` to see how model focus on specific feature of the different paddy image

If you computer the does not have GPU, you can change to CPU by changing the `force_cpu` to True
