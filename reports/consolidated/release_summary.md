# Demo Release Summary

This bundle contains the selected checkpoints, evaluation summaries, and graphics used for the defense/demo package.

## Included branches

### Accident Hybrid H01

- **Task**: accident
- **Role**: current_primary
- **Selection note**: Best held-out accident model in the workspace. Selected as the default UI/inference branch.
- **Checkpoint**: `model_best.pt`

Test metrics:
- `accuracy`: 0.8292682926829268
- `precision`: 0.9090909090909091
- `recall`: 0.7317073170731707
- `f1`: 0.8108108108108109
- `balanced_accuracy`: 0.8292682926829269
- `roc_auc`: 0.9018441403926235
- `pr_auc`: 0.920434345994145

### Accident ConvNeXt D02

- **Task**: accident
- **Role**: best_single_branch
- **Selection note**: Best single-branch ConvNeXt accident line after harmonization and grouped-window splitting.
- **Checkpoint**: `model_best.pt`

Test metrics:
- `accuracy`: 0.8048780487804879
- `precision`: 1.0
- `recall`: 0.6097560975609756
- `f1`: 0.7575757575757576
- `balanced_accuracy`: 0.8048780487804879
- `roc_auc`: 0.8643664485425342
- `pr_auc`: 0.8969191762113444

### Accident A03r Without Harmonization

- **Task**: accident
- **Role**: best_pre_harmonization
- **Selection note**: Best accident line from the non-harmonized branch. Preserved as a reference point for methodology comparison.
- **Checkpoint**: `model_last.pt`

Test metrics:
- `accuracy`: 0.6711111111111111
- `precision`: 0.6037735849056604
- `recall`: 0.9955555555555555
- `f1`: 0.7516778523489933
- `balanced_accuracy`: 0.6711111111111111
- `roc_auc`: 0.9419259259259259
- `pr_auc`: 0.9552274260509066

### Severity S13 R3D

- **Task**: severity
- **Role**: current_primary
- **Selection note**: Current primary severity model. Chosen for the best held-out test quality among the recovered severity runs.
- **Checkpoint**: `model_best.pt`

Test metrics:
- `accuracy`: 0.6444444444444445
- `macro_f1`: 0.6404825577940215
- `balanced_accuracy`: 0.6444444444444445
- `roc_auc_ovr`: 0.8096888888888888

### Severity B01 Baseline

- **Task**: severity
- **Role**: historical_baseline
- **Selection note**: Historical severity baseline. Included to show the gap between the early ConvNeXt branch and the final S13 line.
- **Checkpoint**: `model_last.pt`

Test metrics:
- `accuracy`: 0.3377777777777778
- `macro_f1`: 0.2685301867338756
- `balanced_accuracy`: 0.3377777777777778
- `roc_auc_ovr`: 0.5198666666666667

## Charts

- `reports/consolidated/assets/accident_comparison.png`
- `reports/consolidated/assets/severity_comparison.png`
