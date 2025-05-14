
1. Train vs. Validation Loss + MAE

- train_loss and train_mae steadily decrease — good convergence
- val_loss and val_mae plateau and oscillate after ~5 epochs — clear sign of overfitting

Observation:
- Gap between training and validation error grows → model memorizes training set too closely
- Validation performance stuck around MAE ~9.5–10, while training MAE drops to ~6.5

2. Residual Plot
Residual = Predicted – Actual

Clear trend:

- Young ages (45–60) → overpredicted (positive residuals)
- Older ages (70–80) → underpredicted (negative residuals)

Interpretation:

- The model struggles with age extremes.
- Model regresses toward the mean — common in unbalanced regression datasets.
- Even with WeightedRandomSampler, the model favors dense age regions (~60–70).

Assessment:
- Convergence: Training loss and MAE decreased smoothly; stable and effective learning.
- Validation: Validation loss/MAE plateaued early (~epoch 5), suggesting overfitting.
- Generalization: Weak — model fails to generalize beyond mid-range ages.
- Bias: Model heavily regresses to mean age; performance drops at extremes.

The Baseline model performs adequately for a simple CNN, but does not generalize well — particularly at age extremes.