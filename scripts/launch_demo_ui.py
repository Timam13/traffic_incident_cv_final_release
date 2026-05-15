from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message="'cgi' is deprecated", category=DeprecationWarning)

import cgi
import json
import mimetypes
import os
import shutil
import threading
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from _bootstrap import ensure_src_path

ensure_src_path()

from traffic_incident_cv.inference.saved_run_inference import (
    assess_video_cascade,
    discover_default_run_dir,
    discover_demo_examples,
    write_assessment_bundle,
)
from traffic_incident_cv.utils.io import ensure_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_index_html(*, accident_run_dir: Path | None, severity_run_dir: Path | None, examples: list[dict[str, str]]) -> str:
    examples_html = "".join(
        f"<button class='example-btn' type='button' data-video='{example['video_path']}'>{example['title']}</button>"
        for example in examples
    ) or "<div class='muted'>No examples were found. You can upload your own file.</div>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Traffic Incident CV Demo</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f4efe8;
      --panel: rgba(255,255,255,0.84);
      --ink: #1f1a15;
      --muted: #70665b;
      --line: rgba(31,26,21,0.12);
      --accent: #0b6c58;
      --danger: #a03c22;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Inter", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(11,108,88,0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(160,60,34,0.10), transparent 28%),
        linear-gradient(180deg, #fcf8f2 0%, var(--bg) 100%);
      color: var(--ink);
    }}
    .wrap {{ max-width: 1380px; margin: 0 auto; padding: 28px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 30px;
      box-shadow: 0 18px 46px rgba(49, 38, 28, 0.10);
      backdrop-filter: blur(18px);
    }}
    .hero {{ padding: 28px; margin-bottom: 18px; }}
    .hero h1 {{ margin: 0 0 8px; font-size: 46px; line-height: 1.02; }}
    .hero p {{ margin: 0; color: var(--muted); max-width: 980px; }}
    .grid {{
      display: grid;
      grid-template-columns: 420px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{ padding: 22px; }}
    .label {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      margin-bottom: 12px;
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    input, select, button, textarea {{
      width: 100%;
      font: inherit;
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      background: rgba(255,255,255,0.82);
      color: var(--ink);
    }}
    button {{
      cursor: pointer;
      font-weight: 700;
      background: linear-gradient(135deg, #0c705b 0%, #0e8b6c 100%);
      color: white;
      border: 0;
      box-shadow: 0 14px 24px rgba(11,108,88,0.20);
    }}
    button.secondary {{
      background: rgba(255,255,255,0.70);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }}
    .stack > * + * {{ margin-top: 12px; }}
    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .example-btn {{
      width: auto;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(255,255,255,0.82);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 16px;
      background: rgba(255,255,255,0.72);
    }}
    .card .k {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--muted);
    }}
    .card .v {{
      font-size: 30px;
      font-weight: 700;
      margin-top: 10px;
      line-height: 1.05;
    }}
    .viz-grid {{
      display: grid;
      grid-template-columns: minmax(0, 2fr) minmax(320px, 1fr);
      gap: 14px;
      margin-bottom: 16px;
    }}
    .viz-card {{
      border: 1px solid var(--line);
      border-radius: 24px;
      background: rgba(255,255,255,0.72);
      padding: 12px;
    }}
    .viz-card svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .empty-viz {{
      padding: 18px;
      color: var(--muted);
      font-size: 14px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.11em;
      color: var(--muted);
    }}
    .video-box {{
      border-radius: 24px;
      overflow: hidden;
      background: rgba(31,26,21,0.08);
      border: 1px solid var(--line);
      margin-bottom: 16px;
    }}
    video {{ display: block; width: 100%; max-height: 420px; background: #0f0f0f; }}
    .muted {{ color: var(--muted); }}
    .danger {{ color: var(--danger); }}
    .json {{
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      background: rgba(31,26,21,0.06);
      border-radius: 18px;
      padding: 16px;
      overflow: auto;
      max-height: 360px;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      margin-right: 12px;
    }}
    @media (max-width: 980px) {{
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="label">Demo interface</div>
      <h1>Upload a video for assessment and auto-annotation</h1>
      <p>The interface uses the project's saved models: it first scores accident probability across windows, then predicts severity for positive windows. The output includes a summary, window-level predictions, annotation segments, and an HTML/JSON bundle.</p>
    </section>
    <section class="grid">
      <section class="panel stack">
        <div>
          <div class="label">Configuration</div>
          <label for="accidentRun">Accident model directory</label>
          <input id="accidentRun" value="{accident_run_dir or ''}" />
        </div>
        <div>
          <label for="severityRun">Severity model directory</label>
          <input id="severityRun" value="{severity_run_dir or ''}" />
        </div>
        <div>
          <label for="device">Device</label>
          <select id="device">
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
          </select>
        </div>
        <div>
          <label for="threshold">Accident threshold</label>
          <input id="threshold" type="number" step="0.01" min="0" max="1" value="0.5" />
        </div>
        <div>
          <label for="windowSpan">Window size, frames</label>
          <input id="windowSpan" type="number" min="1" placeholder="auto" />
        </div>
        <div>
          <label for="stepFrames">Window step, frames</label>
          <input id="stepFrames" type="number" min="1" placeholder="auto" />
        </div>
        <div>
          <label for="maxWindows">Limit the number of windows</label>
          <input id="maxWindows" type="number" min="1" placeholder="no limit" />
        </div>
        <div>
          <label for="videoFile">Video</label>
          <input id="videoFile" type="file" accept="video/*" />
        </div>
        <button id="analyzeBtn" type="button">Run assessment</button>
        <button id="clearBtn" class="secondary" type="button">Clear result</button>
        <div>
          <div class="label">Quick examples</div>
          <div class="examples">{examples_html}</div>
        </div>
        <div id="status" class="muted">Waiting for input.</div>
      </section>
      <section class="panel">
        <div class="video-box">
          <video id="preview" controls></video>
        </div>
        <div id="resultCards" class="cards"></div>
        <div class="viz-grid">
          <div class="viz-card">
            <div class="label">Score timeline</div>
            <div id="timelineViz" class="empty-viz">The chart will appear after the assessment finishes.</div>
          </div>
          <div class="viz-card">
            <div class="label">Severity and top windows</div>
            <div id="severityVotesViz" class="empty-viz">The chart will appear after the assessment finishes.</div>
            <div class="label" style="margin-top:14px;">Top windows</div>
            <table>
              <thead>
                <tr>
                  <th>Window</th>
                  <th>Start</th>
                  <th>End</th>
                  <th>Score</th>
                  <th>Severity</th>
                </tr>
              </thead>
              <tbody id="topWindowsBody"><tr><td colspan="5">No data</td></tr></tbody>
            </table>
          </div>
        </div>
        <div class="links" id="artifactLinks"></div>
        <div class="label">Annotation segments</div>
        <table>
          <thead>
            <tr>
              <th>Segment</th>
              <th>Start</th>
              <th>End</th>
              <th>Severity</th>
              <th>Max score</th>
              <th>Windows</th>
            </tr>
          </thead>
          <tbody id="segmentsBody"></tbody>
        </table>
        <div class="label" style="margin-top:18px;">Window predictions</div>
        <table>
          <thead>
            <tr>
              <th>Window</th>
              <th>Start</th>
              <th>End</th>
              <th>Score</th>
              <th>Accident</th>
              <th>Severity</th>
            </tr>
          </thead>
          <tbody id="windowsBody"></tbody>
        </table>
        <div class="label" style="margin-top:18px;">Raw JSON</div>
        <div id="jsonOutput" class="json"></div>
      </section>
    </section>
  </div>
  <script>
    const statusEl = document.getElementById("status");
    const preview = document.getElementById("preview");
    const jsonOutput = document.getElementById("jsonOutput");
    const segmentsBody = document.getElementById("segmentsBody");
    const windowsBody = document.getElementById("windowsBody");
    const resultCards = document.getElementById("resultCards");
    const artifactLinks = document.getElementById("artifactLinks");
    const timelineViz = document.getElementById("timelineViz");
    const severityVotesViz = document.getElementById("severityVotesViz");
    const topWindowsBody = document.getElementById("topWindowsBody");

    function fmt(value, digits = 4) {{
      if (value === null || value === undefined || value === "") return "—";
      const num = Number(value);
      return Number.isFinite(num) ? num.toFixed(digits) : String(value);
    }}

    function safeNumber(value, fallback = 0) {{
      const num = Number(value);
      return Number.isFinite(num) ? num : fallback;
    }}

    function renderCards(summary) {{
      const cards = [
        ["Overall accident label", summary.overall_accident_label],
        ["Max score", fmt(summary.max_accident_score)],
        ["Positive windows", summary.positive_window_count],
        ["Dominant severity", summary.dominant_severity_label || "—"],
        ["Mean score", fmt(summary.mean_accident_score)],
        ["Median score", fmt(summary.median_accident_score)],
        ["Segments", summary.segment_count ?? "—"],
        ["First positive, sec", fmt(summary.first_positive_sec)],
        ["Coverage, sec", fmt(summary.positive_coverage_sec)],
        ["Duration, sec", fmt(summary.duration_sec)],
        ["FPS", fmt(summary.fps, 2)],
      ];
      resultCards.innerHTML = cards.map(([k, v]) => `
        <div class="card">
          <div class="k">${{k}}</div>
          <div class="v">${{v}}</div>
        </div>
      `).join("");
    }}

    function topWindows(windows, limit = 5) {{
      return [...windows].sort((a, b) => safeNumber(b.accident_score) - safeNumber(a.accident_score)).slice(0, limit);
    }}

    function renderTopWindows(windows) {{
      const rows = topWindows(windows);
      topWindowsBody.innerHTML = rows.length ? rows.map((item) => `
        <tr>
          <td>${{item.clip_id}}</td>
          <td>${{fmt(item.start_sec)}}</td>
          <td>${{fmt(item.end_sec)}}</td>
          <td>${{fmt(item.accident_score)}}</td>
          <td>${{item.severity_label || "—"}}</td>
        </tr>
      `).join("") : `<tr><td colspan="5">No data</td></tr>`;
    }}

    function renderTimeline(payload) {{
      const windows = payload.window_predictions || [];
      if (!windows.length) {{
        timelineViz.innerHTML = `<div class="empty-viz">Not enough data to build a timeline.</div>`;
        return;
      }}
      const threshold = safeNumber(payload.summary.accident_threshold, 0.5);
      const duration = Math.max(
        safeNumber(payload.summary.duration_sec, 0),
        ...windows.map((item) => safeNumber(item.end_sec, 0)),
        1
      );
      const width = 1120;
      const height = 270;
      const chartLeft = 72;
      const chartTop = 26;
      const chartWidth = width - 120;
      const chartHeight = 164;
      const scoreBaseY = chartTop + chartHeight;
      const xPos = (sec) => chartLeft + (safeNumber(sec, 0) / duration) * chartWidth;
      const yPos = (score) => scoreBaseY - Math.max(0, Math.min(safeNumber(score, 0), 1)) * chartHeight;

      const grid = [];
      const labels = [];
      for (let tick = 0; tick <= 5; tick += 1) {{
        const ratio = tick / 5;
        const y = chartTop + chartHeight * ratio;
        const value = (1 - ratio).toFixed(1);
        grid.push(`<line x1="${{chartLeft}}" y1="${{y}}" x2="${{chartLeft + chartWidth}}" y2="${{y}}" stroke="rgba(31,26,21,0.10)" stroke-width="1" />`);
        labels.push(`<text x="${{chartLeft - 12}}" y="${{y + 4}}" text-anchor="end" font-size="11" fill="#6f665c">${{value}}</text>`);
      }}
      for (let tick = 0; tick <= 6; tick += 1) {{
        const sec = duration * tick / 6;
        const x = xPos(sec);
        grid.push(`<line x1="${{x}}" y1="${{chartTop}}" x2="${{x}}" y2="${{scoreBaseY}}" stroke="rgba(31,26,21,0.08)" stroke-width="1" />`);
        labels.push(`<text x="${{x}}" y="${{scoreBaseY + 18}}" text-anchor="middle" font-size="11" fill="#6f665c">${{sec.toFixed(1)}}s</text>`);
      }}

      const linePoints = [];
      const areaPoints = [`${{chartLeft}},${{scoreBaseY}}`];
      const circles = [];
      windows.forEach((item) => {{
        const mid = (safeNumber(item.start_sec) + safeNumber(item.end_sec)) / 2;
        const score = safeNumber(item.accident_score);
        const x = xPos(mid);
        const y = yPos(score);
        linePoints.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
        areaPoints.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
        const fill = score >= threshold ? "#a03c22" : "#0d6b55";
        circles.push(`<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="5" fill="${{fill}}" stroke="white" stroke-width="2" />`);
      }});
      areaPoints.push(`${{chartLeft + chartWidth}},${{scoreBaseY}}`);

      const bandY = scoreBaseY + 32;
      const bands = (payload.segments || []).map((segment) => {{
        const startX = xPos(segment.start_sec);
        const endX = xPos(segment.end_sec);
        const bandWidth = Math.max(endX - startX, 4);
        const severity = segment.severity_label || "detected";
        const color = severity === "major" ? "#a03c22" : severity === "moderate" ? "#c17e17" : "#0d6b55";
        return `
          <rect x="${{startX.toFixed(2)}}" y="${{bandY.toFixed(2)}}" width="${{bandWidth.toFixed(2)}}" height="16" rx="8" fill="${{color}}" fill-opacity="0.92" />
          <text x="${{(startX + bandWidth / 2).toFixed(2)}}" y="${{(bandY + 12).toFixed(2)}}" text-anchor="middle" font-size="10" fill="white">${{severity}}</text>
        `;
      }}).join("");

      const thresholdY = yPos(threshold);
      timelineViz.innerHTML = `
        <svg viewBox="0 0 1120 270" role="img" aria-label="Score timeline by window">
          <defs>
            <linearGradient id="scoreGradientUi" x1="0" x2="0" y1="0" y2="1">
              <stop offset="0%" stop-color="#0d6b55" stop-opacity="0.34"></stop>
              <stop offset="100%" stop-color="#0d6b55" stop-opacity="0.03"></stop>
            </linearGradient>
          </defs>
          <rect x="0" y="0" width="${{width}}" height="${{height}}" rx="24" fill="rgba(255,255,255,0.72)"></rect>
          ${{grid.join("")}}
          ${{labels.join("")}}
          <line x1="${{chartLeft}}" y1="${{thresholdY.toFixed(2)}}" x2="${{chartLeft + chartWidth}}" y2="${{thresholdY.toFixed(2)}}" stroke="#a03c22" stroke-width="2" stroke-dasharray="6 6" />
          <text x="${{chartLeft + chartWidth - 4}}" y="${{(thresholdY - 8).toFixed(2)}}" text-anchor="end" font-size="11" fill="#a03c22">threshold ${{threshold.toFixed(2)}}</text>
          <polygon points="${{areaPoints.join(" ")}}" fill="url(#scoreGradientUi)"></polygon>
          <polyline fill="none" stroke="#0d6b55" stroke-width="3" points="${{linePoints.join(" ")}}"></polyline>
          ${{circles.join("")}}
          ${{bands}}
          <text x="${{chartLeft}}" y="${{(bandY + 38).toFixed(2)}}" font-size="11" fill="#6f665c">Annotation segments</text>
        </svg>
      `;
    }}

    function renderSeverityVotes(payload) {{
      const votes = payload.severity_vote_counts || {{}};
      const labels = Object.entries(votes).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
      if (!labels.length) {{
        severityVotesViz.innerHTML = `<div class="empty-viz">No positive windows were found, so the severity chart is unavailable.</div>`;
        return;
      }}
      const maxValue = Math.max(...labels.map(([, value]) => Number(value)));
      const width = 420;
      const height = 210;
      const chartLeft = 96;
      const chartTop = 30;
      const barHeight = 28;
      const barGap = 18;
      const bars = labels.map(([label, value], index) => {{
        const barWidth = (Number(value) / maxValue) * (width - chartLeft - 36);
        const y = chartTop + index * (barHeight + barGap);
        const color = label === "major" ? "#a03c22" : label === "moderate" ? "#c17e17" : "#0d6b55";
        return `
          <text x="18" y="${{(y + 19).toFixed(2)}}" font-size="13" fill="#6f665c">${{label}}</text>
          <rect x="${{chartLeft}}" y="${{y.toFixed(2)}}" width="${{barWidth.toFixed(2)}}" height="${{barHeight}}" rx="14" fill="${{color}}" />
          <text x="${{(chartLeft + barWidth + 10).toFixed(2)}}" y="${{(y + 19).toFixed(2)}}" font-size="13" fill="#1c1814">${{value}}</text>
        `;
      }}).join("");
      severityVotesViz.innerHTML = `
        <svg viewBox="0 0 420 210" role="img" aria-label="Severity vote counts">
          <rect x="0" y="0" width="420" height="210" rx="22" fill="rgba(255,255,255,0.72)"></rect>
          <text x="18" y="22" font-size="12" fill="#6f665c" style="letter-spacing:0.14em;text-transform:uppercase;">Severity votes</text>
          ${{bars}}
        </svg>
      `;
    }}

    function renderRows(payload) {{
      const segments = payload.segments || [];
      const windows = payload.window_predictions || [];
      segmentsBody.innerHTML = segments.length ? segments.map((item) => `
        <tr>
          <td>${{item.segment_id}}</td>
          <td>${{fmt(item.start_sec)}}</td>
          <td>${{fmt(item.end_sec)}}</td>
          <td>${{item.severity_label || "—"}}</td>
          <td>${{fmt(item.max_accident_score)}}</td>
          <td>${{item.window_count}}</td>
        </tr>
      `).join("") : `<tr><td colspan="6">No segments detected</td></tr>`;
      windowsBody.innerHTML = windows.map((item) => `
        <tr>
          <td>${{item.clip_id}}</td>
          <td>${{fmt(item.start_sec)}}</td>
          <td>${{fmt(item.end_sec)}}</td>
          <td>${{fmt(item.accident_score)}}</td>
          <td>${{item.accident_label || "—"}}</td>
          <td>${{item.severity_label || "—"}}</td>
        </tr>
      `).join("");
    }}

    function renderArtifacts(bundlePaths) {{
      const links = [];
      if (bundlePaths?.report_html) {{
        links.push(`<a href="/files?path=${{encodeURIComponent(bundlePaths.report_html)}}" target="_blank">HTML report</a>`);
      }}
      if (bundlePaths?.assessment_json) {{
        links.push(`<a href="/files?path=${{encodeURIComponent(bundlePaths.assessment_json)}}" target="_blank">assessment.json</a>`);
      }}
      if (bundlePaths?.annotations_jsonl) {{
        links.push(`<a href="/files?path=${{encodeURIComponent(bundlePaths.annotations_jsonl)}}" target="_blank">annotations.jsonl</a>`);
      }}
      artifactLinks.innerHTML = links.join("");
    }}

    async function runAnalysis(formData) {{
      statusEl.textContent = "Assessment started...";
      statusEl.className = "muted";
      const response = await fetch("/api/analyze", {{ method: "POST", body: formData }});
      const payload = await response.json();
      if (!response.ok) {{
        statusEl.textContent = payload.error || "Error";
        statusEl.className = "danger";
        jsonOutput.textContent = JSON.stringify(payload, null, 2);
        return;
      }}
      statusEl.textContent = "Done.";
      renderCards(payload.result.summary);
      renderTimeline(payload.result);
      renderSeverityVotes(payload.result);
      renderTopWindows(payload.result.window_predictions || []);
      renderRows(payload.result);
      renderArtifacts(payload.bundle_paths);
      jsonOutput.textContent = JSON.stringify(payload, null, 2);
      if (payload.bundle_paths?.copied_video_path) {{
        preview.src = `/files?path=${{encodeURIComponent(payload.bundle_paths.copied_video_path)}}`;
      }}
    }}

    document.getElementById("videoFile").addEventListener("change", (event) => {{
      const file = event.target.files?.[0];
      if (!file) return;
      preview.src = URL.createObjectURL(file);
      statusEl.textContent = `Selected file: ${{file.name}}`;
    }});

    document.getElementById("analyzeBtn").addEventListener("click", async () => {{
      const video = document.getElementById("videoFile").files?.[0];
      if (!video) {{
        statusEl.textContent = "Select a video file.";
        statusEl.className = "danger";
        return;
      }}
      const formData = new FormData();
      formData.append("video", video);
      formData.append("accident_run_dir", document.getElementById("accidentRun").value);
      formData.append("severity_run_dir", document.getElementById("severityRun").value);
      formData.append("device", document.getElementById("device").value);
      formData.append("accident_threshold", document.getElementById("threshold").value || "0.5");
      formData.append("window_span_frames", document.getElementById("windowSpan").value || "");
      formData.append("step_frames", document.getElementById("stepFrames").value || "");
      formData.append("max_windows", document.getElementById("maxWindows").value || "");
      await runAnalysis(formData);
    }});

    document.getElementById("clearBtn").addEventListener("click", () => {{
      statusEl.textContent = "Waiting for input.";
      statusEl.className = "muted";
      resultCards.innerHTML = "";
      artifactLinks.innerHTML = "";
      segmentsBody.innerHTML = "";
      windowsBody.innerHTML = "";
      topWindowsBody.innerHTML = `<tr><td colspan="5">No data</td></tr>`;
      timelineViz.innerHTML = `<div class="empty-viz">The chart will appear after the assessment finishes.</div>`;
      severityVotesViz.innerHTML = `<div class="empty-viz">The chart will appear after the assessment finishes.</div>`;
      jsonOutput.textContent = "";
      preview.removeAttribute("src");
      preview.load();
      document.getElementById("videoFile").value = "";
    }});

    document.querySelectorAll(".example-btn").forEach((button) => {{
      button.addEventListener("click", async () => {{
        const formData = new FormData();
        formData.append("example_video_path", button.dataset.video);
        formData.append("accident_run_dir", document.getElementById("accidentRun").value);
        formData.append("severity_run_dir", document.getElementById("severityRun").value);
        formData.append("device", document.getElementById("device").value);
        formData.append("accident_threshold", document.getElementById("threshold").value || "0.5");
        formData.append("window_span_frames", document.getElementById("windowSpan").value || "");
        formData.append("step_frames", document.getElementById("stepFrames").value || "");
        formData.append("max_windows", document.getElementById("maxWindows").value || "");
        preview.src = `/files?path=${{encodeURIComponent(button.dataset.video)}}`;
        await runAnalysis(formData);
      }});
    }});
  </script>
</body>
</html>
"""


class DemoRequestHandler(BaseHTTPRequestHandler):
    accident_run_dir: Path | None = None
    severity_run_dir: Path | None = None
    output_root: Path
    examples: list[dict[str, str]]
    lock = threading.Lock()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, body: str, status: int = 200) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(
                _build_index_html(
                    accident_run_dir=self.accident_run_dir,
                    severity_run_dir=self.severity_run_dir,
                    examples=self.examples,
                )
            )
            return
        if parsed.path == "/files":
            params = parse_qs(parsed.query)
            target_path = params.get("path", [""])[0]
            if not target_path:
                self._send_json({"error": "Missing path"}, status=400)
                return
            path = Path(target_path).resolve()
            allowed_roots = [self.output_root.resolve(), PROJECT_ROOT.resolve()]
            if not any(str(path).startswith(str(root)) for root in allowed_roots):
                self._send_json({"error": "Path is outside allowed roots"}, status=403)
                return
            if not path.exists():
                self._send_json({"error": "File not found"}, status=404)
                return
            data = path.read_bytes()
            mime_type, _ = mimetypes.guess_type(path.name)
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send_json({"error": "Not found"}, status=404)
            return
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type", ""),
            },
        )
        try:
            with self.lock:
                run_id = time_safe_id()
                run_dir = ensure_dir(self.output_root / "runs" / run_id)
                upload_path = None
                example_video_path = form.getfirst("example_video_path")
                if example_video_path:
                    upload_path = Path(example_video_path).resolve()
                else:
                    video_field = form["video"] if "video" in form else None
                    if video_field is None or not getattr(video_field, "filename", None):
                        self._send_json({"error": "Video file is required"}, status=400)
                        return
                    upload_path = run_dir / Path(video_field.filename).name
                    with upload_path.open("wb") as fout:
                        shutil.copyfileobj(video_field.file, fout)

                accident_run_dir = Path(form.getfirst("accident_run_dir") or self.accident_run_dir or "")
                severity_value = form.getfirst("severity_run_dir") or str(self.severity_run_dir or "")
                severity_run_dir = Path(severity_value).resolve() if severity_value else None
                if not accident_run_dir or not accident_run_dir.exists():
                    self._send_json({"error": "The accident model directory was not found or is invalid"}, status=400)
                    return
                if severity_run_dir is not None and not severity_run_dir.exists():
                    severity_run_dir = None
                threshold = float(form.getfirst("accident_threshold") or 0.5)
                window_span = _optional_int(form.getfirst("window_span_frames"))
                step_frames = _optional_int(form.getfirst("step_frames"))
                max_windows = _optional_int(form.getfirst("max_windows"))
                device = form.getfirst("device") or "cuda"

                result = assess_video_cascade(
                    video_path=upload_path,
                    accident_run_dir=accident_run_dir,
                    severity_run_dir=severity_run_dir,
                    device_name=device,
                    accident_threshold=threshold,
                    window_span_frames=window_span,
                    step_frames=step_frames,
                    max_windows=max_windows,
                    temp_dir=run_dir / "_temp",
                )
                bundle_paths = write_assessment_bundle(
                    output_dir=run_dir,
                    payload=result,
                    uploaded_video_path=upload_path,
                )
                self._send_json(
                    {
                        "run_id": run_id,
                        "bundle_paths": bundle_paths,
                        "result": result,
                    }
                )
        except Exception as exc:  # pragma: no cover
            self._send_json({"error": str(exc)}, status=500)


def _optional_int(value: str | None) -> int | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return int(stripped)


def time_safe_id() -> str:
    return uuid.uuid4().hex[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch local UI for video upload and model-based assessment.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output-root", default="outputs/demo_ui")
    parser.add_argument("--accident-run-dir")
    parser.add_argument("--severity-run-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = ensure_dir(PROJECT_ROOT / args.output_root)
    accident_run_dir = (
        Path(args.accident_run_dir).resolve()
        if args.accident_run_dir
        else discover_default_run_dir(PROJECT_ROOT, "accident")
    )
    severity_run_dir = (
        Path(args.severity_run_dir).resolve()
        if args.severity_run_dir
        else discover_default_run_dir(PROJECT_ROOT, "severity")
    )
    DemoRequestHandler.output_root = output_root
    DemoRequestHandler.accident_run_dir = accident_run_dir
    DemoRequestHandler.severity_run_dir = severity_run_dir
    DemoRequestHandler.examples = discover_demo_examples(PROJECT_ROOT)
    server = ThreadingHTTPServer((args.host, args.port), DemoRequestHandler)
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{args.port}",
                "output_root": str(output_root.resolve()),
                "accident_run_dir": str(accident_run_dir) if accident_run_dir else None,
                "severity_run_dir": str(severity_run_dir) if severity_run_dir else None,
                "example_count": len(DemoRequestHandler.examples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
