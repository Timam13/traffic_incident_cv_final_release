from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from _bootstrap import ensure_src_path

ensure_src_path()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_REGISTRY = PROJECT_ROOT / "assets" / "models" / "selected_models.json"
DEFAULT_PROJECT_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

DEFAULT_TRAIN_CONFIGS = {
    "accident": "configs/experiments/accident_convnext_balanced_grouped_windows_gap96_real_roadcrop_rtx3080_10gb.yaml",
    "severity": "configs/experiments/severity_r3d_uniform_pod.yaml",
}

TRAIN_SCRIPT_BY_TASK = {
    "accident": "train_accident.py",
    "severity": "train_severity.py",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_path(project_root: Path, raw_path: str | None) -> Path | None:
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def _load_registry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model registry was not found: {path}")
    payload = _load_json(path)
    if "models" not in payload or not isinstance(payload["models"], dict):
        raise ValueError(f"Invalid model registry format: {path}")
    return payload


def _resolve_model_dir(registry: dict[str, Any], name: str) -> Path | None:
    model_entry = registry.get("models", {}).get(name, {})
    return _resolve_path(PROJECT_ROOT, model_entry.get("path"))


def _model_status(registry: dict[str, Any], name: str) -> dict[str, Any]:
    model_entry = registry.get("models", {}).get(name, {})
    model_dir = _resolve_model_dir(registry, name)
    required_files = list(model_entry.get("required_files", []))
    existing_required = []
    missing_required = []
    if model_dir is not None:
        for filename in required_files:
            file_path = model_dir / filename
            if file_path.exists():
                existing_required.append(str(file_path))
            else:
                missing_required.append(str(file_path))
    return {
        "name": name,
        "description": model_entry.get("description"),
        "configured_path": model_entry.get("path"),
        "resolved_path": str(model_dir) if model_dir is not None else None,
        "exists": bool(model_dir and model_dir.exists()),
        "required_ok": len(missing_required) == 0,
        "existing_required_files": existing_required,
        "missing_required_files": missing_required,
    }


def _run_python_script(script_name: str, args: list[str]) -> int:
    script_path = PROJECT_ROOT / "scripts" / script_name
    cmd = [str(_resolve_python_executable()), str(script_path), *args]
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def _resolve_python_executable() -> Path:
    explicit = os.getenv("TRAFFIC_INCIDENT_PYTHON")
    if explicit:
        explicit_path = _resolve_path(PROJECT_ROOT, explicit)
        if explicit_path is not None and explicit_path.exists() and _python_is_usable(explicit_path):
            return explicit_path
    if DEFAULT_PROJECT_PYTHON.exists() and _python_is_usable(DEFAULT_PROJECT_PYTHON):
        return DEFAULT_PROJECT_PYTHON
    return Path(sys.executable)


def _python_is_usable(python_path: Path) -> bool:
    try:
        result = subprocess.run(
            [str(python_path), "-V"],
            cwd=PROJECT_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except Exception:
        return False
    return result.returncode == 0


def cmd_info(args: argparse.Namespace) -> int:
    registry = _load_registry(Path(args.model_registry).resolve())
    payload = {
        "project_root": str(PROJECT_ROOT),
        "python_executable": str(_resolve_python_executable()),
        "model_registry": str(Path(args.model_registry).resolve()),
        "ui_entry_point": registry.get("ui_entry_point"),
        "models": {
            "accident": _model_status(registry, "accident"),
            "severity": _model_status(registry, "severity"),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def cmd_ui(args: argparse.Namespace) -> int:
    registry = _load_registry(Path(args.model_registry).resolve())
    accident_dir = Path(args.accident_run_dir).resolve() if args.accident_run_dir else _resolve_model_dir(registry, "accident")
    severity_dir = Path(args.severity_run_dir).resolve() if args.severity_run_dir else _resolve_model_dir(registry, "severity")
    cli_args = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--output-root",
        args.output_root,
    ]
    if accident_dir is not None:
        cli_args.extend(["--accident-run-dir", str(accident_dir)])
    if severity_dir is not None:
        cli_args.extend(["--severity-run-dir", str(severity_dir)])
    return _run_python_script("launch_demo_ui.py", cli_args)


def cmd_train(args: argparse.Namespace) -> int:
    script_name = TRAIN_SCRIPT_BY_TASK[args.task]
    config_path = args.config or DEFAULT_TRAIN_CONFIGS[args.task]
    cli_args = ["--config", config_path]
    if args.dry_run:
        cli_args.append("--dry-run")
    return _run_python_script(script_name, cli_args)


def cmd_infer(args: argparse.Namespace) -> int:
    registry = _load_registry(Path(args.model_registry).resolve())
    accident_dir = Path(args.accident_run_dir).resolve() if args.accident_run_dir else _resolve_model_dir(registry, "accident")
    severity_dir = Path(args.severity_run_dir).resolve() if args.severity_run_dir else _resolve_model_dir(registry, "severity")
    if accident_dir is None:
        raise FileNotFoundError("No accident model directory resolved. Set --accident-run-dir or fix assets registry.")
    cli_args = [
        "--video",
        args.video,
        "--device",
        args.device,
        "--accident-threshold",
        str(args.accident_threshold),
        "--accident-run-dir",
        str(accident_dir),
    ]
    if severity_dir is not None:
        cli_args.extend(["--severity-run-dir", str(severity_dir)])
    if args.window_span_frames is not None:
        cli_args.extend(["--window-span-frames", str(args.window_span_frames)])
    if args.step_frames is not None:
        cli_args.extend(["--step-frames", str(args.step_frames)])
    if args.max_windows is not None:
        cli_args.extend(["--max-windows", str(args.max_windows)])
    if args.output_dir:
        cli_args.extend(["--output-dir", args.output_dir])
    return _run_python_script("infer_saved_video.py", cli_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single management entry point for train/test/info/ui/infer flows.",
    )
    parser.add_argument(
        "--model-registry",
        default=str(DEFAULT_MODEL_REGISTRY),
        help="Path to JSON model registry with selected best checkpoints.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_parser = subparsers.add_parser("info", help="Show resolved model registry status.")
    info_parser.set_defaults(func=cmd_info)

    ui_parser = subparsers.add_parser("ui", help="Launch demo UI using selected models.")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8090)
    ui_parser.add_argument("--output-root", default="outputs/demo_ui")
    ui_parser.add_argument("--accident-run-dir")
    ui_parser.add_argument("--severity-run-dir")
    ui_parser.set_defaults(func=cmd_ui)

    train_parser = subparsers.add_parser("train", help="Train one task via existing train scripts.")
    train_parser.add_argument("task", choices=sorted(TRAIN_SCRIPT_BY_TASK))
    train_parser.add_argument("--config", help="Override config path.")
    train_parser.add_argument("--dry-run", action="store_true")
    train_parser.set_defaults(func=cmd_train)
    

    infer_parser = subparsers.add_parser("infer", help="Infer on one saved video with selected models.")
    infer_parser.add_argument("--video", required=True)
    infer_parser.add_argument("--accident-run-dir")
    infer_parser.add_argument("--severity-run-dir")
    infer_parser.add_argument("--device", default="cuda")
    infer_parser.add_argument("--accident-threshold", type=float, default=0.5)
    infer_parser.add_argument("--window-span-frames", type=int)
    infer_parser.add_argument("--step-frames", type=int)
    infer_parser.add_argument("--max-windows", type=int)
    infer_parser.add_argument("--output-dir")
    infer_parser.set_defaults(func=cmd_infer)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    exit_code = args.func(args)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
