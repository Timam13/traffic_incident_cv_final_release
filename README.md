# Traffic Incident CV Demo Release

Computer vision pipeline for road-incident detection and accident-severity classification in roadway video streams. The project is organized as a reproducible research/demo repository: it contains source code, configurations, manifests, selected lightweight model artifacts, demo videos, consolidated reports, and scripts for training, evaluation, inference, and Web UI execution.

The implemented system follows a cascade:

```text
input video
   -> window / clip extraction
   -> accident detector: accident / no_accident
   -> if accident is detected: severity classifier
   -> severity: minor / moderate / major
```

The repository focuses on three implemented branches:

| Branch                      | Folder                            | Role                                                                             |
| --------------------------- | --------------------------------- | -------------------------------------------------------------------------------- |
| H01 hybrid accident model   | `models/hybrid/`                | Main accident model for the demo UI and aggregate accident-detection quality.    |
| D02 ConvNeXt accident model | `models/accident_convnext_d02/` | Conservative single-branch accident detector with strict false-positive control. |
| S13 R3D severity model      | `models/severity/`              | Main severity classifier for `minor / moderate / major`.                       |

Two additional model folders are kept as historical comparison artifacts:

| Branch | Folder                                     | Role                                             |
| ------ | ------------------------------------------ | ------------------------------------------------ |
| A03r   | `models/accident_a03r_no_harmonization/` | Pre-harmonization accident reference branch.     |
| B01    | `models/severity_b01_baseline/`          | Early severity baseline for comparison with S13. |

---

## 1. Repository contents

```text
.
├── assets/
├── configs/
├── data/
├── demo_data/
├── models/
├── reports/
├── scripts/
├── src/
├── .env.example
├── pyproject.toml
├── requirements.txt
├── requirements-train.txt
├── setup_venv.cmd
├── launch_ui.cmd
├── launch_ui_public.cmd
└── README.md
```

### Folder meaning

| Path                         | Meaning                                                                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `assets/`                  | Small registry/configuration assets used by management and UI scripts.                                                                                                               |
| `configs/`                 | YAML experiment and dataset configs. This is the main configuration layer for training/evaluation.                                                                                   |
| `data/`                    | Lightweight data manifests. Raw datasets and large generated videos are not stored here by default.                                                                                  |
| `demo_data/`               | Small curated demo bundle for the Web UI: short videos, UI examples, and reference assessment reports. Demo videos are intentionally part of the repository.                         |
| `models/`                  | Lightweight model-release folders: configs, metrics, summaries, reports, visualizations, and expected checkpoint names.`.pt` weights may be stored locally or supplied externally. |
| `reports/`                 | Consolidated summary reports and comparison figures.                                                                                                                                 |
| `scripts/`                 | Command-line entry points for training, evaluation, inference, manifest preparation, and UI launch.                                                                                  |
| `src/traffic_incident_cv/` | Main Python package: data processing, models, evaluation, inference, pipelines, and utilities.                                                                                       |
| `outputs/`                 | Runtime/generated outputs. This directory is created during training, evaluation, inference, or UI use.                                                                              |

---

## 2. Model checkpoint files (`.pt`)

Large PyTorch checkpoint files may be omitted from GitHub because of file-size limits. The repository still keeps the expected checkpoint filenames in model metadata.

| Model                       | Expected local checkpoint path                          |
| --------------------------- | ------------------------------------------------------- |
| H01 hybrid accident model   | `models/hybrid/model_best.pt`                         |
| D02 ConvNeXt accident model | `models/accident_convnext_d02/model_best.pt`          |
| S13 R3D severity model      | `models/severity/model_best.pt`                       |
| A03r reference model        | `models/accident_a03r_no_harmonization/model_last.pt` |
| B01 baseline model          | `models/severity_b01_baseline/model_last.pt`          |

Link for weighted model: [https://disk.360.yandex.ru/d/oUwCJtSdcKwZHg](https://disk.360.yandex.ru/d/oUwCJtSdcKwZHg)

---

## 3. Environment setup

### Python version

The project is designed for Python `3.10+`. Python `3.12` is suitable. If Python `3.13+` is used and an error appears around `import cgi`, install `legacy-cgi` or ensure it is present in `pyproject.toml` dependencies.

### Windows setup

From the repository root:

```bat
setup_venv.cmd
```

Manual equivalent:

```bat
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

### macOS / Linux setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

### Basic install without training extras

```bash
python -m pip install -r requirements.txt
```

### Training / full CV install

```bash
python -m pip install -r requirements-train.txt
```

If GPU training is required, install a PyTorch build compatible with your CUDA version before or during the environment setup.

---

## 4. Running the Web UI

### Windows, local UI

```bat
launch_ui.cmd
```

Equivalent command:

```bat
python scripts\manage.py ui --host 127.0.0.1 --port 8090 --output-root outputs\demo_ui
```

Open:

```text
http://127.0.0.1:8090
```

### Windows, public/local-network UI

```bat
launch_ui_public.cmd
```

This starts the UI with:

```text
host = 0.0.0.0
port = 8090
```

Use this only when you intentionally want access from another device on the same network.

### macOS / Linux UI

```bash
source .venv/bin/activate
python scripts/manage.py ui --host 127.0.0.1 --port 8090 --output-root outputs/demo_ui
```

For access from the local network:

```bash
python scripts/manage.py ui --host 0.0.0.0 --port 8090 --output-root outputs/demo_ui
```

---

## 5. Running inference on a saved video

Use the selected model folders automatically:

```bash
python scripts/manage.py infer \
  --video demo_data/videos/positive/accident_major_defense_reference.mp4 \
  --device cpu \
  --output-dir outputs/manual_inference
```

Or specify model folders explicitly:

```bash
python scripts/infer_saved_video.py \
  --video demo_data/videos/positive/accident_major_defense_reference.mp4 \
  --accident-run-dir models/hybrid \
  --severity-run-dir models/severity \
  --device cpu \
  --output-dir outputs/manual_inference
```

Use `--device cuda` if CUDA and compatible PyTorch are available.

---

## 6. Training commands

Training requires local datasets/manifests and may require model checkpoints, raw videos, or generated harmonized videos depending on the selected config.

### Train D02 accident ConvNeXt branch

```bash
python scripts/train_accident.py \
  --config configs/experiments/accident_convnext_balanced_grouped_windows_gap96_real_roadcrop_rtx3080_10gb.yaml
```

### Train H01 hybrid accident branch

```bash
python scripts/train_accident.py \
  --config configs/experiments/accident_hybrid_grouped_windows_gap96_real_rtx3080_10gb.yaml
```

### Train S13 severity R3D branch

```bash
python scripts/train_severity.py \
  --config configs/experiments/severity_r3d_uniform_pod.yaml
```

### Dry run

```bash
python scripts/train_accident.py \
  --config configs/experiments/accident_convnext_balanced_grouped_windows_gap96_real_roadcrop_rtx3080_10gb.yaml \
  --dry-run
```

---

## 7. Demo data

`demo_data/` contains a compact curated UI bundle:

```text
demo_data/
├── ui_examples.json
├── videos/
│   ├── positive/
│   └── negative/
└── assessments/
```

This folder is intentionally small and suitable for a GitHub demo. It is not a replacement for the full training datasets.

To rebuild the demo bundle from harmonized videos and current models:

```bash
python scripts/build_ui_test_dataset.py
```

This requires harmonized source videos under:

```text
outputs/harmonized_videos/bridge_v1_8fps_160/
```

and usable model checkpoints.

---

## 8. How to duplicate this Git repository correctly

### Option: clone the repository

```bash
git clone <REPOSITORY_URL>
cd <REPOSITORY_FOLDER>
```

Then create the environment:

```bash
python -m venv .venv
source .venv/bin/activate       # macOS/Linux
# or .venv\Scripts\activate.bat # Windows
python -m pip install --upgrade pip
python -m pip install -e ".[train]"
```

### Before pushing a cleaned copy

Remove generated cache files:

```bash
find . -type d -name "__pycache__" -prune -exec rm -rf {} +
find . -type d -name "__MACOSX" -prune -exec rm -rf {} +
find . -type f \( -name ".DS_Store" -o -name "._*" -o -name "*.pyc" -o -name "*.pyo" \) -delete
```

Check what will be committed:

```bash
git status
```

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'cgi'`

Install `legacy-cgi`:

```bash
python -m pip install legacy-cgi
```

or ensure that `legacy-cgi` is listed in `pyproject.toml` dependencies.

### `ModuleNotFoundError: No module named 'cv2'`

Install training/CV dependencies:

```bash
python -m pip install -e ".[train]"
```

### PyTorch / CUDA problems

Install a PyTorch build matching your CUDA version. If GPU is unavailable, run commands with:

```bash
--device cpu
```

### UI starts but model inference fails

Check that checkpoint files exist at the expected paths, especially:

```text
models/hybrid/model_best.pt
models/severity/model_best.pt
models/accident_convnext_d02/model_best.pt
```

---

## 10. License and reuse

Suggested project license for code: MIT License, unless a separate `LICENSE` file states otherwise.

Datasets are not relicensed by this repository. AI City, Balanced Accident Video Dataset, and any other external datasets remain subject to their original licenses and terms of use.

If this repository, its model artifacts, or its experimental structure are used in further academic or applied research, cite the work as:

```text
Mamedov, T. (2026). Computer Vision for Automated Detection of Traffic Incidents and Infrastructure-Related Anomalies in Roadway Video Streams. Bachelor Thesis / Research Project.
```

BibTeX placeholder:

```bibtex
@misc{mamedov2026trafficincidentcv,
  author       = {Mamedov, Timur},
  title        = {Computer Vision for Automated Detection of Traffic Incidents and Infrastructure-Related Anomalies in Roadway Video Streams},
  year         = {2026},
  howpublished = {Bachelor thesis / research project},
  note         = {Traffic incident computer vision pipeline for accident detection and severity classification}
}
```

---

## 11. Current scope

Implemented:

```text
accident / no_accident detection
minor / moderate / major severity classification
hybrid full-frame + local evidence accident branch
saved-run evaluation
Web UI demo
curated demo video bundle
```

Future extension:

```text
infrastructure anomaly detection
long-stream operational false-alarm analysis
```
