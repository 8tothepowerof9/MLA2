1. To run the code
   To train the code run the train_model.py script. For example: python train_model.py

To evaluate the model, run eval_model.py script. For example: python eval_model.py

To train a model type, change the model architecture in train_model.py, and change the loss function and augmentation as well. For example, CBAM Resnet 18 Focal loss uses the CBAMResNet18 (in cbam.py) and change the loss to focal loss. The model weights are automatically saved.

After that, simply change the model architecture and weights path in eval_model.py to evaluate the model.

2. Other files explaination
   The data folder contains the data to train the model. The dataloader.py contains the code to load the data, it has options such as oversample which enables oversampling in the data. The trainer.py file contains the code for the trainer which train the models. By default, it doesn't use mixup/cutmix, but you can make it uses these augmentations by setting mixup=True. The models folder contains the code for each models: CBAM variations models, resnet model (resnet18) and the independent evaluation model (paper.py). hyper.py handle the hyperparameter tuning.  

Remember to add the data folder in the variety folder to train
