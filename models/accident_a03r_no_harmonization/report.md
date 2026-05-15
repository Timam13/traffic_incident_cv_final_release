# Accident A03r Without Harmonization

- **Task**: accident
- **Role in release**: best_pre_harmonization
- **Checkpoint file**: `model_last.pt`
- **Source run**: `outputs/A03r_accident_convnext_clipneg_bridge_v1_fixed_rtx3080_10gb`
- **Selection note**: Best accident line from the non-harmonized branch. Preserved as a reference point for methodology comparison.

## Test metrics

- **accuracy**: 0.6711
- **precision**: 0.6038
- **recall**: 0.9956
- **f1**: 0.7517
- **balanced_accuracy**: 0.6711
- **roc_auc**: 0.9419
- **pr_auc**: 0.9552

## Confusion matrix

- `[[78, 147], [1, 224]]`

## Visual assets

- `test_confusion_matrix.png`
- `test_threshold_sweep.png`
