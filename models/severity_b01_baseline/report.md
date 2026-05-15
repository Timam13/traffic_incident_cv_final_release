# Severity B01 Baseline

- **Task**: severity
- **Role in release**: historical_baseline
- **Checkpoint file**: `model_last.pt`
- **Source run**: `outputs/_tmp_b01_localized`
- **Selection note**: Historical severity baseline. Included to show the gap between the early ConvNeXt branch and the final S13 line.

## Test metrics

- **accuracy**: 0.3378
- **macro_f1**: 0.2685
- **balanced_accuracy**: 0.3378
- **roc_auc_ovr**: 0.5199

## Confusion matrix

- `[[0, 34, 41], [0, 32, 43], [0, 31, 44]]`

## Visual assets

- `test_confusion_matrix.png`
- `test_margin_histogram.png`
