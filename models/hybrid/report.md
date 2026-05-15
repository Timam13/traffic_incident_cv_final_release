# Accident Hybrid H01

- **Task**: accident
- **Role in release**: current_primary
- **Checkpoint file**: `model_best.pt`
- **Source run**: `outputs/H01_accident_hybrid_grouped_windows_gap96_real_rtx3080_10gb`
- **Selection note**: Best held-out accident model in the workspace. Selected as the default UI/inference branch.

## Test metrics

- **accuracy**: 0.8293
- **precision**: 0.9091
- **recall**: 0.7317
- **f1**: 0.8108
- **balanced_accuracy**: 0.8293
- **roc_auc**: 0.9018
- **pr_auc**: 0.9204

## Confusion matrix

- `[[38, 3], [11, 30]]`

## Visual assets

- `test_confusion_matrix.png`
- `test_threshold_sweep.png`
