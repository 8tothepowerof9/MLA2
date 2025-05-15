# Dataset Folder – Rice Disease Classification

This folder contains all image datasets and metadata used throughout the pipeline, including original images, lesion-masked variants, test images, and U-Net outputs. It supports both the **main classification task** and **literature review experiments**.

## Original Data
Used for the main task of the assignment
`train_images/` Original training images with labeled rice diseases.                      
`test_images/`  Images for submission (labels unknown).     
`meta_train.csv` Metadata CSV linking `train_images/` to disease labels and other info.  

### Experimental & Processed Data for Literature review
Used for literature review and comparisons between experimented model with research papers on the same problem. 
#### `generated_masks/`
- Contains **binary lesion masks** predicted by the **U-Net model** for each training image.
- Format: PNG images where white regions indicate predicted lesion areas.
- Used as an **intermediate step** before creating masked input images.

#### `masked_images/`
- Final training images after **applying U-Net masks** to the original inputs (mask × image).
- Used in the **masked training experiment** to emphasize lesion regions.
- Corresponds 1-to-1 with `train_images/`.
#### `rice_images/`
- A **different rice dataset** used in the **literature review**, which was the closed up version of the crops for comparative evaluation between datasets with and without lesion-based masking.
#### `masked_rice_images/`
- Masked images generated from the above rice_images, used for the proposed masked model with the new dataset. This data generate with the Unet model trained in the original data, and will then be fed into the fine tuned version of the 2 models developed to compare. 

### `unet/` – Manual Annotations for Segmentation

- Contains **manually labeled images** used to train the lesion segmentation U-Net.
- Each disease class is represented by **10 randomly selected images** (i.e., balanced subset).
- Manual masks were created using **LabelMe**, a polygon-based image annotation tool.
- These labeled masks were converted to binary format for training the U-Net model in a supervised fashion.
## Masking Workflow
1. Input `train_images/`  
2. ➡ Predict lesion using U-Net  
3. ➡ Save to `generated_masks/`  
4. ➡ Multiply mask × image → `masked_images/`



