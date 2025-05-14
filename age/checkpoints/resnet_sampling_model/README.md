1. Loss Curve (train/val)
- Train loss steadily declines, showing good convergence.

- Val loss decreases with noise/spikes, but no overfitting is evident.

- Final val loss is significantly lower than the baseline.

Training is stable and successful. Generalization is better than the baseline. Spikes may come from label imbalance or batch noise despite weighted sampling.

2. MAE Curve (train/val)
- Train MAE improves down to ~3.2
- Val MAE improves from ~8.2 to ~4.5, with fluctuations

Compared to baseline:

- Baseline final val MAE: ~9.5
- ResNet final val MAE: ~4.5

Assessment:

- ResNet halves the error compared to baseline.
- Even with fluctuations, it's clearly more powerful and generalizes better.


3. R² Score Curve
- Train R² improves up to ~0.85
- Val R² improves to ~0.58, despite fluctuations

R² > 0.5 on validation means:

- Model captures meaningful variance in target
- Still has room to improve in explaining full age variance

Assessment:

R² shows consistent learning and outperforms baseline significantly (which had val R² ~0.1–0.2)

4. Residual Plot
- Less extreme residuals at both ends than baseline.

- Still shows slight underprediction for older ages and overprediction for younger ages, but:
    - Bias is reduced
    - Spread is narrower
    - Residuals are better centered around 0

Assessment:
- ResNet reduces regression-to-mean bias.
- Model is more confident and more balanced across age groups.
- Weighted sampling appears more effective here.

ResNet with weighted sampling is a clear improvement over the baseline.
It learns better features, generalizes better, and predicts more accurately across all age groups.
Fluctuations in validation metrics are expected with smaller datasets and are manageable.