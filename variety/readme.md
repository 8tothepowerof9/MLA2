This folder contains the training and evaluation code for the variety class.

1. Project structure
2. 
checkpoints contain the training history and model weights.

data contains the training and labels of the dataset

models contain the code of the models and custom layers I tested with for this assignment. This includes:
- paper.py: Model from paper research
- cbam.py: Cbam implementation as well as integration into the resnet18 model
- resnet.py: Model that uses resnet as the backbone, this have the option to use pretrained weights or not
- baseline.py: Simple CNN model with some special layers to reduce overfitting

dataloader.py contains the code to load the dataset form data folder. The minority classes are oversampled to 1000 instances, while the majority classes are undersampled down to 1000 classes

eda.ipynb contains the code to explore the data of this task

eval_model.py contains the code to evaluate models. To use it, simply change the model architecture in the code directly, and change the path to the model weights, and uncomment the functions you want to use

train_model.py contains the code to train the models. Similar to eval_model.py, to use it, simply change the model architecture and loss function you want to use. The model weights and training history will be saved by default.

trainer.py contains the code of the trainer, which is used in train_model.py to train models. By default, the trainer doesn't use mixup/cutmix, but you can set it to use this augmentation by setting mixup=True when creating the Trainer instance

hyper.py contains the code to do hyperparameter tuning on the model. After tuning, it will plot the tuning history as well as printing the best combination of parameters.

2. How to run

To run:
- `cd variety`
- `python train_model.py`
- `python eval_model.py`
- `python hyper.py`

Make sure you have the complete training data in the data folder, or else it won't be able to laod the data.
