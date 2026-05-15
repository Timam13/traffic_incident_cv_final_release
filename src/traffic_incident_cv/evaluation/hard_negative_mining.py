from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from traffic_incident_cv.data.manifest import save_manifest
from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.utils.io import ensure_dir, write_json, write_jsonl


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def _binary_label(value: Any) -> str | None:
    if hasattr(value, 'value'):
        return str(value.value)
    if value is None:
        return None
    return str(value)


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, float(q))) * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_score_distribution(values: Iterable[float]) -> dict[str, float | None]:
    scores = [float(value) for value in values]
    if not scores:
        return {'count': 0, 'min': None, 'p50': None, 'p90': None, 'p95': None, 'p99': None, 'max': None}
    return {
        'count': len(scores),
        'min': min(scores),
        'p50': _quantile(scores, 0.5),
        'p90': _quantile(scores, 0.9),
        'p95': _quantile(scores, 0.95),
        'p99': _quantile(scores, 0.99),
        'max': max(scores),
        'mean': sum(scores) / len(scores),
        'median': median(scores),
    }


def join_records_with_predictions(records: list[ClipRecord], prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_predictions = {str(row.get('clip_id')): row for row in prediction_rows}
    joined: list[dict[str, Any]] = []
    for record in records:
        prediction = indexed_predictions.get(record.clip_id)
        if prediction is None:
            continue
        metadata = prediction.get('metadata', {})
        joined.append(
            {
                'clip_id': record.clip_id,
                'score': float(prediction.get('score') or 0.0),
                'target_label': _binary_label(record.binary_label),
                'predicted_label': str(prediction.get('predicted_label')) if prediction.get('predicted_label') is not None else None,
                'target_index': metadata.get('target_index'),
                'predicted_index': metadata.get('predicted_index'),
                'camera_id': record.camera_id,
                'dataset': record.dataset,
                'video_path': record.video_path,
                'clip_start_sec': record.clip_start_sec,
                'clip_end_sec': record.clip_end_sec,
                'clip_start_frame': record.clip_start_frame,
                'clip_end_frame': record.clip_end_frame,
                'tags': list(record.tags),
            }
        )
    return joined


def select_hard_negatives(
    joined_rows: list[dict[str, Any]],
    *,
    decision_threshold: float,
    selection_score_floor: float,
    top_k: int,
) -> dict[str, Any]:
    negatives = [row for row in joined_rows if row.get('target_label') == 'no_accident']
    positives = [row for row in joined_rows if row.get('target_label') == 'accident']
    false_positives = [row for row in negatives if float(row.get('score') or 0.0) >= decision_threshold]
    scored_negatives = sorted(negatives, key=lambda item: float(item.get('score') or 0.0), reverse=True)
    floor_candidates = [row for row in scored_negatives if float(row.get('score') or 0.0) >= selection_score_floor]
    if top_k > 0:
        selected = floor_candidates[:top_k] if floor_candidates else scored_negatives[:top_k]
    else:
        selected = floor_candidates if floor_candidates else list(scored_negatives)

    def _group_top(rows: list[dict[str, Any]], key: str, limit: int = 12) -> list[dict[str, Any]]:
        counter = Counter(str(row.get(key) or '<none>') for row in rows)
        return [{key: name, 'count': count} for name, count in counter.most_common(limit)]

    return {
        'negative_score_distribution': summarize_score_distribution(row['score'] for row in negatives),
        'positive_score_distribution': summarize_score_distribution(row['score'] for row in positives),
        'false_positive_count_at_threshold': len(false_positives),
        'selection_candidate_count': len(floor_candidates),
        'selected_count': len(selected),
        'selected': selected,
        'top_false_positive_cameras': _group_top(false_positives, 'camera_id'),
        'top_selected_cameras': _group_top(selected, 'camera_id'),
        'top_selected_videos': _group_top(selected, 'video_path'),
    }


def build_hard_negative_manifest(
    records: list[ClipRecord],
    *,
    selected_rows: list[dict[str, Any]],
    repeat_factor: int,
    source_run_id: str,
) -> tuple[list[ClipRecord], list[dict[str, Any]]]:
    selected_index = {str(row.get('clip_id')): row for row in selected_rows}
    base_records = list(records)
    appended_records: list[ClipRecord] = []
    appended_rows: list[dict[str, Any]] = []
    hard_negative_rank = {str(row.get('clip_id')): rank for rank, row in enumerate(selected_rows, start=1)}
    for record in records:
        selected = selected_index.get(record.clip_id)
        if selected is None:
            continue
        for copy_index in range(1, max(int(repeat_factor), 0) + 1):
            new_tags = list(record.tags)
            new_tags.extend(
                [
                    'hard_negative:mined',
                    f'hard_negative_source:{source_run_id}',
                    f'hard_negative_rank:{hard_negative_rank[record.clip_id]}',
                    f"hard_negative_score:{float(selected.get('score') or 0.0):.4f}",
                    f'hard_negative_copy:{copy_index}',
                ]
            )
            appended = ClipRecord(
                clip_id=f'{record.clip_id}__hardneg_mined_r{copy_index:02d}',
                video_path=record.video_path,
                dataset=record.dataset,
                split=record.split,
                task=record.task,
                binary_label=record.binary_label,
                severity_label=record.severity_label,
                camera_id=record.camera_id,
                fps=record.fps,
                duration_sec=record.duration_sec,
                clip_start_sec=record.clip_start_sec,
                clip_end_sec=record.clip_end_sec,
                clip_start_frame=record.clip_start_frame,
                clip_end_frame=record.clip_end_frame,
                tags=new_tags,
            )
            appended_records.append(appended)
            appended_rows.append(
                {
                    'clip_id': appended.clip_id,
                    'source_clip_id': record.clip_id,
                    'score': float(selected.get('score') or 0.0),
                    'camera_id': record.camera_id,
                    'video_path': record.video_path,
                    'copy_index': copy_index,
                    'rank': hard_negative_rank[record.clip_id],
                    'tags': new_tags,
                }
            )
    return [*base_records, *appended_records], appended_rows


def write_hard_negative_bundle(
    *,
    output_dir: str | Path,
    manifest_output_path: str | Path,
    source_run_id: str,
    split: str,
    decision_threshold: float,
    selection_score_floor: float,
    repeat_factor: int,
    joined_rows: list[dict[str, Any]],
    mining_summary: dict[str, Any],
    mined_records: list[ClipRecord],
    appended_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    target_dir = ensure_dir(output_dir)
    manifest_path = Path(manifest_output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    save_manifest(manifest_path, mined_records)

    selected_rows = list(mining_summary.get('selected', []))
    selected_jsonl = target_dir / f'{split}_hard_negative_selected.jsonl'
    candidates_jsonl = target_dir / f'{split}_hard_negative_candidates.jsonl'
    appended_jsonl = target_dir / f'{split}_hard_negative_appended_rows.jsonl'
    summary_json = target_dir / f'{split}_hard_negative_summary.json'
    report_html = target_dir / f'{split}_hard_negative_report.html'

    write_jsonl(candidates_jsonl, joined_rows)
    write_jsonl(selected_jsonl, selected_rows)
    write_jsonl(appended_jsonl, appended_rows)

    original_count = len(joined_rows)
    original_negative_count = sum(1 for row in joined_rows if row.get('target_label') == 'no_accident')
    original_positive_count = sum(1 for row in joined_rows if row.get('target_label') == 'accident')
    appended_count = len(appended_rows)
    mined_negative_count = original_negative_count + appended_count
    payload = {
        'source_run_id': source_run_id,
        'split': split,
        'decision_threshold': decision_threshold,
        'selection_score_floor': selection_score_floor,
        'repeat_factor': repeat_factor,
        'original_count': original_count,
        'original_negative_count': original_negative_count,
        'original_positive_count': original_positive_count,
        'appended_hard_negative_count': appended_count,
        'mined_count': len(mined_records),
        'mined_negative_count': mined_negative_count,
        'mined_positive_count': original_positive_count,
        'negative_uplift_ratio': (mined_negative_count / original_negative_count) if original_negative_count else None,
        **{key: value for key, value in mining_summary.items() if key != 'selected'},
        'manifest_output_path': str(manifest_path),
        'selected_rows_path': str(selected_jsonl),
        'candidates_rows_path': str(candidates_jsonl),
        'appended_rows_path': str(appended_jsonl),
    }
    write_json(summary_json, payload)

    selected_rows_preview = selected_rows[:20]
    top_cameras = payload.get('top_selected_cameras', [])
    negative_uplift_ratio = payload.get('negative_uplift_ratio') or 0.0
    html = f'''<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<title>Hard Negative Mining Report</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f6f1e7;color:#241f1a;margin:0;padding:24px;}}
.wrap{{max-width:1280px;margin:0 auto;}}
.hero{{background:#fffaf2;border:1px solid #dacbb2;border-radius:20px;padding:24px 28px;box-shadow:0 12px 40px rgba(61,43,25,.08);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-top:18px;}}
.card{{background:#fff;border:1px solid #e4d8c3;border-radius:16px;padding:16px;}}
h1,h2{{margin:0 0 12px 0;}}
small{{color:#6d6255;}}
table{{width:100%;border-collapse:collapse;margin-top:12px;background:#fff;border-radius:14px;overflow:hidden;}}
th,td{{padding:10px 12px;border-bottom:1px solid #efe5d6;text-align:left;font-size:14px;}}
th{{background:#f3eadb;}}
code{{font-family:Consolas,monospace;font-size:12px;}}
.section{{margin-top:22px;}}
pre{{background:#1f1b17;color:#f5eee5;padding:14px;border-radius:14px;overflow:auto;}}
</style>
</head>
<body>
<div class="wrap">
  <div class="hero">
    <small>TRAFFIC INCIDENT CV HARD NEGATIVE MINING</small>
    <h1>{source_run_id} / {split}</h1>
    <div>Manifest: <code>{manifest_path}</code></div>
    <div class="grid">
      <div class="card"><small>Original clips</small><div style="font-size:34px;font-weight:700;">{original_count}</div></div>
      <div class="card"><small>Selected hard negatives</small><div style="font-size:34px;font-weight:700;">{payload['selected_count']}</div></div>
      <div class="card"><small>Extra copies</small><div style="font-size:34px;font-weight:700;">{appended_count}</div></div>
      <div class="card"><small>Mined clips</small><div style="font-size:34px;font-weight:700;">{payload['mined_count']}</div></div>
      <div class="card"><small>FP @ {decision_threshold:.2f}</small><div style="font-size:34px;font-weight:700;">{payload['false_positive_count_at_threshold']}</div></div>
      <div class="card"><small>Negative uplift</small><div style="font-size:34px;font-weight:700;">{negative_uplift_ratio:.2f}x</div></div>
    </div>
  </div>

  <div class="section">
    <h2>Score Distribution</h2>
    <pre>{json.dumps({'negative': payload.get('negative_score_distribution', {}), 'positive': payload.get('positive_score_distribution', {})}, ensure_ascii=False, indent=2)}</pre>
  </div>

  <div class="section">
    <h2>Top Cameras</h2>
    <table><thead><tr><th>camera_id</th><th>count</th></tr></thead><tbody>
      {''.join(f"<tr><td>{row.get('camera_id')}</td><td>{row.get('count')}</td></tr>" for row in top_cameras)}
    </tbody></table>
  </div>

  <div class="section">
    <h2>Selected Hard Negatives Preview</h2>
    <table><thead><tr><th>clip_id</th><th>score</th><th>camera_id</th><th>window</th></tr></thead><tbody>
      {''.join(f"<tr><td><code>{row.get('clip_id')}</code></td><td>{float(row.get('score') or 0.0):.4f}</td><td>{row.get('camera_id')}</td><td>{row.get('clip_start_frame')}..{row.get('clip_end_frame')}</td></tr>" for row in selected_rows_preview)}
    </tbody></table>
  </div>
</div>
</body>
</html>
'''
    report_html.write_text(html, encoding='utf-8')
    return {
        'summary_json': str(summary_json),
        'report_html': str(report_html),
        'manifest_output_path': str(manifest_path),
        'selected_rows_path': str(selected_jsonl),
        'appended_rows_path': str(appended_jsonl),
    }
