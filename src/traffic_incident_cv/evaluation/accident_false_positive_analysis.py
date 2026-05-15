from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from traffic_incident_cv.data.manifest import load_manifest
from traffic_incident_cv.domain.schemas import ClipRecord
from traffic_incident_cv.evaluation.hard_negative_mining import join_records_with_predictions, read_prediction_rows
from traffic_incident_cv.utils.io import ensure_dir, write_json


def _group_rows(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or '<none>')].append(row)
    return dict(grouped)


def _summarize_group(false_positive_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]], key: str, *, top_k: int = 15) -> list[dict[str, Any]]:
    negative_groups = _group_rows(negative_rows, key)
    fp_groups = _group_rows(false_positive_rows, key)
    rows: list[dict[str, Any]] = []
    for name, items in fp_groups.items():
        scores = [float(item.get('score') or 0.0) for item in items]
        total_negatives = len(negative_groups.get(name, []))
        rows.append(
            {
                key: name,
                'false_positive_count': len(items),
                'negative_count': total_negatives,
                'fp_rate': (len(items) / total_negatives) if total_negatives else None,
                'mean_score': mean(scores) if scores else None,
                'max_score': max(scores) if scores else None,
            }
        )
    rows.sort(key=lambda item: (item['false_positive_count'], item.get('fp_rate') or 0.0, item.get('max_score') or 0.0), reverse=True)
    return rows[:top_k]


def _summarize_tag_groups(false_positive_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]], *, top_k: int = 20) -> list[dict[str, Any]]:
    negative_counter: Counter[str] = Counter()
    fp_counter: Counter[str] = Counter()
    for row in negative_rows:
        for tag in row.get('tags', []):
            negative_counter[str(tag)] += 1
    for row in false_positive_rows:
        for tag in row.get('tags', []):
            fp_counter[str(tag)] += 1
    rows: list[dict[str, Any]] = []
    for tag, fp_count in fp_counter.most_common(top_k):
        negative_count = negative_counter.get(tag, 0)
        rows.append(
            {
                'tag': tag,
                'false_positive_count': fp_count,
                'negative_count': negative_count,
                'fp_rate': (fp_count / negative_count) if negative_count else None,
            }
        )
    return rows


def _top_rows(rows: list[dict[str, Any]], *, top_k: int = 25) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda item: float(item.get('score') or 0.0), reverse=True)
    preview: list[dict[str, Any]] = []
    for row in ranked[:top_k]:
        preview.append(
            {
                'clip_id': row.get('clip_id'),
                'camera_id': row.get('camera_id'),
                'dataset': row.get('dataset'),
                'score': float(row.get('score') or 0.0),
                'video_path': row.get('video_path'),
                'clip_start_sec': row.get('clip_start_sec'),
                'clip_end_sec': row.get('clip_end_sec'),
                'tags': list(row.get('tags', [])),
            }
        )
    return preview


def build_false_positive_analysis(
    *,
    records: list[ClipRecord],
    prediction_rows: list[dict[str, Any]],
    run_name: str,
    split: str,
    threshold: float,
) -> dict[str, Any]:
    joined_rows = join_records_with_predictions(records, prediction_rows)
    negative_rows = [row for row in joined_rows if row.get('target_label') == 'no_accident']
    positive_rows = [row for row in joined_rows if row.get('target_label') == 'accident']
    false_positive_rows = [row for row in negative_rows if float(row.get('score') or 0.0) >= threshold]
    false_negative_rows = [row for row in positive_rows if float(row.get('score') or 0.0) < threshold]
    return {
        'run_name': run_name,
        'split': split,
        'threshold': threshold,
        'record_count': len(joined_rows),
        'negative_count': len(negative_rows),
        'positive_count': len(positive_rows),
        'false_positive_count': len(false_positive_rows),
        'false_negative_count': len(false_negative_rows),
        'false_positive_rate': (len(false_positive_rows) / len(negative_rows)) if negative_rows else None,
        'false_negative_rate': (len(false_negative_rows) / len(positive_rows)) if positive_rows else None,
        'camera_breakdown': _summarize_group(false_positive_rows, negative_rows, 'camera_id'),
        'video_breakdown': _summarize_group(false_positive_rows, negative_rows, 'video_path'),
        'dataset_breakdown': _summarize_group(false_positive_rows, negative_rows, 'dataset', top_k=8),
        'tag_breakdown': _summarize_tag_groups(false_positive_rows, negative_rows),
        'top_false_positive_rows': _top_rows(false_positive_rows),
    }


def compare_false_positive_analyses(analyses: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = list(analyses)
    if len(ordered) < 2:
        return {'runs': [item.get('run_name') for item in ordered], 'delta': {}}
    base = ordered[0]
    candidate = ordered[1]
    return {
        'runs': [base.get('run_name'), candidate.get('run_name')],
        'delta': {
            'false_positive_count': int(candidate.get('false_positive_count', 0)) - int(base.get('false_positive_count', 0)),
            'false_positive_rate': (candidate.get('false_positive_rate') or 0.0) - (base.get('false_positive_rate') or 0.0),
            'false_negative_count': int(candidate.get('false_negative_count', 0)) - int(base.get('false_negative_count', 0)),
        },
    }


def write_false_positive_report(
    *,
    output_dir: str | Path,
    analyses: list[dict[str, Any]],
    comparison: dict[str, Any],
) -> dict[str, str]:
    target_dir = ensure_dir(output_dir)
    json_path = target_dir / 'accident_false_positive_analysis.json'
    html_path = target_dir / 'accident_false_positive_analysis.html'
    payload = {
        'analyses': analyses,
        'comparison': comparison,
    }
    write_json(json_path, payload)
    data_json = json.dumps(payload, ensure_ascii=False)
    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Accident False Positive Analysis</title>
<style>
body{{font-family:Segoe UI,Arial,sans-serif;background:#f5efe5;color:#231d17;margin:0;padding:24px;}}
.wrap{{max-width:1380px;margin:0 auto;display:grid;gap:18px;}}
.hero,.panel{{background:#fffaf2;border:1px solid #d9ccb4;border-radius:20px;padding:22px 24px;box-shadow:0 14px 36px rgba(63,44,21,.08);}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;}}
.card{{background:#fff;border:1px solid #e4d8c3;border-radius:16px;padding:14px;}}
button{{border:1px solid #cdbd9e;background:#fff;border-radius:999px;padding:10px 16px;cursor:pointer;}}
button.active{{background:#1e5f52;color:#fff;border-color:#1e5f52;}}
table{{width:100%;border-collapse:collapse;background:#fff;border-radius:14px;overflow:hidden;}}
th,td{{padding:10px 12px;border-bottom:1px solid #efe6d8;text-align:left;vertical-align:top;font-size:14px;}}
small,.muted{{color:#6d6255;}}
</style>
</head>
<body>
<div class="wrap">
  <section class="hero">
    <small>Traffic Incident CV</small>
    <h1>False Positive Analysis</h1>
    <p class="muted">Comparison of runs by false alarms on negative clips.</p>
    <div id="comparison-cards" class="grid"></div>
    <div id="run-buttons" style="display:flex;gap:10px;flex-wrap:wrap;margin-top:16px;"></div>
  </section>
  <section class="panel">
    <h2 id="run-title">Run</h2>
    <div id="run-cards" class="grid"></div>
  </section>
  <section class="panel">
    <h2>Top Cameras</h2>
    <div id="camera-table"></div>
  </section>
  <section class="panel">
    <h2>Top Videos</h2>
    <div id="video-table"></div>
  </section>
  <section class="panel">
    <h2>Tag Breakdown</h2>
    <div id="tag-table"></div>
  </section>
  <section class="panel">
    <h2>Top False Positive Clips</h2>
    <div id="row-table"></div>
  </section>
</div>
<script>
const payload = {data_json};
const analyses = payload.analyses || [];
const comparison = payload.comparison || {{}};
function fmt(value, digits = 4) {{
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
  return Number(value).toFixed(digits);
}}
function renderCards(target, rows) {{
  target.innerHTML = rows.map((row) => `<div class="card"><small>${{row.label}}</small><div style="font-size:34px;font-weight:700;margin-top:6px;">${{row.value}}</div><div class="muted">${{row.note || ''}}</div></div>`).join('');
}}
function renderTable(target, rows, columns) {{
  if (!rows.length) {{
    target.innerHTML = '<p class="muted">No data available.</p>';
    return;
  }}
  const header = columns.map((col) => `<th>${{col.label}}</th>`).join('');
  const body = rows.map((row) => `<tr>${{columns.map((col) => `<td>${{col.render ? col.render(row[col.key], row) : (row[col.key] ?? '')}}</td>`).join('')}}</tr>`).join('');
  target.innerHTML = `<table><thead><tr>${{header}}</tr></thead><tbody>${{body}}</tbody></table>`;
}}
function renderRun(index) {{
  const analysis = analyses[index];
  document.getElementById('run-title').textContent = `${{analysis.run_name}} | split=${{analysis.split}} | threshold=${{analysis.threshold}}`;
  renderCards(document.getElementById('run-cards'), [
    {{ label: 'False Positives', value: String(analysis.false_positive_count), note: 'count of negatives predicted as accident' }},
    {{ label: 'FP Rate', value: fmt(analysis.false_positive_rate), note: 'false_positive_count / negative_count' }},
    {{ label: 'False Negatives', value: String(analysis.false_negative_count), note: 'count of positives missed by threshold' }},
    {{ label: 'Negative Clips', value: String(analysis.negative_count), note: 'available negatives on split' }},
  ]);
  renderTable(document.getElementById('camera-table'), analysis.camera_breakdown || [], [
    {{ key: 'camera_id', label: 'camera_id' }},
    {{ key: 'false_positive_count', label: 'FP' }},
    {{ key: 'negative_count', label: 'Negatives' }},
    {{ key: 'fp_rate', label: 'FP rate', render: (value) => fmt(value) }},
    {{ key: 'mean_score', label: 'Mean score', render: (value) => fmt(value) }},
    {{ key: 'max_score', label: 'Max score', render: (value) => fmt(value) }},
  ]);
  renderTable(document.getElementById('video-table'), analysis.video_breakdown || [], [
    {{ key: 'video_path', label: 'video_path' }},
    {{ key: 'false_positive_count', label: 'FP' }},
    {{ key: 'negative_count', label: 'Negatives' }},
    {{ key: 'fp_rate', label: 'FP rate', render: (value) => fmt(value) }},
    {{ key: 'mean_score', label: 'Mean score', render: (value) => fmt(value) }},
  ]);
  renderTable(document.getElementById('tag-table'), analysis.tag_breakdown || [], [
    {{ key: 'tag', label: 'tag' }},
    {{ key: 'false_positive_count', label: 'FP' }},
    {{ key: 'negative_count', label: 'Negatives' }},
    {{ key: 'fp_rate', label: 'FP rate', render: (value) => fmt(value) }},
  ]);
  renderTable(document.getElementById('row-table'), analysis.top_false_positive_rows || [], [
    {{ key: 'clip_id', label: 'clip_id' }},
    {{ key: 'camera_id', label: 'camera_id' }},
    {{ key: 'score', label: 'score', render: (value) => fmt(value) }},
    {{ key: 'clip_start_sec', label: 'start_sec', render: (value) => fmt(value, 2) }},
    {{ key: 'clip_end_sec', label: 'end_sec', render: (value) => fmt(value, 2) }},
    {{ key: 'video_path', label: 'video_path' }},
  ]);
  Array.from(document.querySelectorAll('#run-buttons button')).forEach((button, buttonIndex) => {{
    button.classList.toggle('active', buttonIndex === index);
  }});
}}
function init() {{
  renderCards(document.getElementById('comparison-cards'), [
    {{ label: 'Base run', value: (comparison.runs || [])[0] || 'n/a', note: 'comparison reference' }},
    {{ label: 'Candidate run', value: (comparison.runs || [])[1] || 'n/a', note: 'comparison target' }},
    {{ label: 'Delta FP', value: String((comparison.delta || {{}}).false_positive_count ?? 'n/a'), note: 'candidate - base' }},
    {{ label: 'Delta FP rate', value: fmt((comparison.delta || {{}}).false_positive_rate), note: 'candidate - base' }},
  ]);
  const buttons = document.getElementById('run-buttons');
  buttons.innerHTML = analyses.map((analysis, index) => `<button data-index="${{index}}">${{analysis.run_name}}</button>`).join('');
  Array.from(buttons.querySelectorAll('button')).forEach((button) => {{
    button.addEventListener('click', () => renderRun(Number(button.dataset.index || 0)));
  }});
  if (analyses.length) {{
    renderRun(0);
  }}
}}
init();
</script>
</body>
</html>
'''
    html_path.write_text(html, encoding='utf-8')
    return {
        'json_path': str(json_path),
        'html_path': str(html_path),
    }


def load_analysis_inputs(predictions_path: str | Path, resolved_manifest_path: str | Path) -> tuple[list[dict[str, Any]], list[ClipRecord]]:
    prediction_rows = read_prediction_rows(predictions_path)
    records = load_manifest(resolved_manifest_path)
    return prediction_rows, records
