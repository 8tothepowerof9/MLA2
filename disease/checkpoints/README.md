# Model Checkpoints and Experiment History
This folder stores all saved model weights and training history logs from experiments related to rice disease classification. It includes both baseline and experimental models, as well as training metrics for performance analysis.
### Model Files (.pt)

- `resnet34_cbam.pt`: The baseline ResNet34 model with CBAM, trained on the full (unmasked) dataset.
- `resnet34_cbam_masked.pt`: Trained on lesion-masked field images to emphasize disease regions.
- `resnet34_cbam_masked_new_data.pt`: Fine-tuned model on a new close-up dataset using masked input.
- `resnet34_cbam_finetuned_new_data.pt`: Fine-tuned model on a new close-up dataset using normal input.

### Training Histories (.json) — Located in 

- `history_cbam_34.json`: Training/validation accuracy and loss log for the baseline model.
- `history_cbam_34_masked.json`: History for the masked version trained on the field dataset.
- `history_finetune_new_data.json`: Fine-tuning history for adapting to the close-up dataset.
- `history_resnet34_cbam_masked_new_data.json`: Retraining history on the new data using masked input.
These history files can be used to plot training curves and compare experiment outcomes by running the utils.inspectors file. 
## `unet/` – U-Net for Lesion Masking

This directory contains the segmentation model trained to isolate disease lesions in rice leaves. This was used for **experimental comparison** in the literature review section.
### Subfolders
- `visualizations/`: Contains the training history logs plotted and confusion matrices of different .
- `unet/`: Reserved for storing segmentation model checkpoints (e.g., U-Net). Currently collapsed in view.

## Loading Example

```python
import torch
from torchvision import models

# Load classification model
model = torch.load("checkpoints/model_cbam.pt")
```




