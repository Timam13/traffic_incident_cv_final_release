from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from traffic_incident_cv.domain.schemas import ExperimentConfig


ENV_PATTERN = re.compile(r"\$\{([^}:]+)(?::-([^}]*))?\}")


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    return _expand_env_vars(payload)


def _expand_env_vars(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return ENV_PATTERN.sub(_replace_env_var, value)
    return value


def _replace_env_var(match: re.Match[str]) -> str:
    env_name = match.group(1)
    default = match.group(2)
    return os.environ.get(env_name, default if default is not None else match.group(0))


def merge_dicts(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path, *, _seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    seen = set() if _seen is None else set(_seen)
    if config_path in seen:
        chain = " -> ".join(str(path) for path in [*seen, config_path])
        raise ValueError(f"Config inheritance cycle detected: {chain}")
    seen.add(config_path)

    config = read_yaml(config_path)
    merged: dict[str, Any] = {}
    inherits = config.get("inherits", [])
    for inherited in inherits:
        inherited_path = _resolve_inherited_path(config_path, inherited)
        merged = merge_dicts(merged, load_config(inherited_path, _seen=seen))
    merged = merge_dicts(merged, {k: v for k, v in config.items() if k != "inherits"})
    return merged


def _resolve_inherited_path(config_path: Path, inherited: str | Path) -> Path:
    inherited_path = Path(inherited)
    if inherited_path.is_absolute():
        return inherited_path.resolve()
    local_candidate = (config_path.parent / inherited_path).resolve()
    if local_candidate.exists():
        return local_candidate
    return (Path.cwd() / inherited_path).resolve()


def to_experiment_config(config: dict[str, Any]) -> ExperimentConfig:
    experiment = config.get("experiment", {})
    return ExperimentConfig(
        experiment_id=experiment["id"],
        task=experiment["task"],
        manifest_paths=experiment.get("manifest_paths", {}),
        model=config.get("model", {}),
        training=config.get("training", {}),
        runtime=config.get("runtime", {}),
        data=config.get("data", {}),
        project=config.get("project", {}),
        evaluation=config.get("evaluation", {}),
        optimization=config.get("optimization", {}),
        reporting=config.get("reporting", {}),
    )
