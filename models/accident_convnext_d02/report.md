# Accident ConvNeXt D02

- **Task**: accident
- **Role in release**: best_single_branch
- **Checkpoint file**: `model_best.pt`
- **Source run**: `outputs/D02_accident_convnext_balanced_grouped_windows_gap96_real_roadcrop_rtx3080_10gb`
- **Selection note**: Best single-branch ConvNeXt accident line after harmonization and grouped-window splitting.

## Test metrics

- **accuracy**: 0.8049
- **precision**: 1.0000
- **recall**: 0.6098
- **f1**: 0.7576
- **balanced_accuracy**: 0.8049
- **roc_auc**: 0.8644
- **pr_auc**: 0.8969

## Confusion matrix

- `[[41, 0], [16, 25]]`

## Visual assets

- `test_confusion_matrix.png`
- `test_threshold_sweep.png`
