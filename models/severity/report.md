# Severity S13 R3D

- **Task**: severity
- **Role in release**: current_primary
- **Checkpoint file**: `model_best.pt`
- **Source run**: `outputs/runpod_severity_push/S13_severity_r3d_uniform_pod`
- **Selection note**: Current primary severity model. Chosen for the best held-out test quality among the recovered severity runs.

## Test metrics

- **accuracy**: 0.6444
- **macro_f1**: 0.6405
- **balanced_accuracy**: 0.6444
- **roc_auc_ovr**: 0.8097

## Confusion matrix

- `[[45, 20, 10], [24, 39, 12], [6, 8, 61]]`

## Visual assets

- `test_confusion_matrix.png`
- `test_margin_histogram.png`
