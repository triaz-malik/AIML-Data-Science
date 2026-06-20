# Phase 8 — Model Comparison (held-out validation set)

Positive class = **damage**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| cnn | 0.8706 | 0.8596 | 0.8860 | 0.8726 | 0.9335 |
| resnet50 | 0.9605 | 0.9487 | 0.9737 | 0.9610 | 0.9924 |
| efficientnet_b0 | 0.9627 | 0.9648 | 0.9605 | 0.9626 | 0.9915 |

**Best model by F1: `efficientnet_b0` (F1=0.9626, accuracy=0.9627, ROC-AUC=0.9915).**

## Figures

![ROC](figures/roc_curves.png)
![PR](figures/pr_curves.png)
![confusion cnn](figures/confusion_cnn.png)
![confusion resnet50](figures/confusion_resnet50.png)
![confusion efficientnet_b0](figures/confusion_efficientnet_b0.png)