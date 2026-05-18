from __future__ import annotations

import argparse
import warnings

warnings.filterwarnings("ignore", message="'cgi' is deprecated", category=DeprecationWarning)

import cgi
import json
import mimetypes
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
      --panel: rgba(255, 255, 255, 0.86);
      --panel-strong: rgba(255, 255, 255, 0.94);
      --ink: #1f1a15;
      --muted: #71685f;
      --soft: #9b9288;
      --line: rgba(31, 26, 21, 0.13);
      --line-strong: rgba(31, 26, 21, 0.22);
      --accent: #0b6c58;
      --accent-2: #0e8b6c;
      --danger: #a03c22;
      --warning: #c17e17;
      --shadow: 0 18px 46px rgba(49, 38, 28, 0.10);
      --radius-lg: 30px;
      --radius-md: 22px;
      --radius-sm: 16px;
    }}

    * {{
      box-sizing: border-box;
    }}

    body {{
      margin: 0;
      font-family: Inter, "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background:
        radial-gradient(circle at top left, rgba(11, 108, 88, 0.12), transparent 34%),
        radial-gradient(circle at top right, rgba(160, 60, 34, 0.10), transparent 28%),
        linear-gradient(180deg, #fcf8f2 0%, var(--bg) 100%);
      color: var(--ink);
    }}

    .wrap {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 28px;
    }}

    .hero,
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius-lg);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }}

    .hero {{
      padding: 26px 28px;
      margin-bottom: 18px;
    }}

    .eyebrow,
    .section-title,
    .field-label,
    .card .k,
    th,
    .mini-head {{
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.13em;
      font-weight: 750;
    }}

    .eyebrow {{
      font-size: 13px;
      margin-bottom: 8px;
    }}

    .hero h1 {{
      margin: 0 0 10px;
      font-size: clamp(34px, 4vw, 52px);
      line-height: 0.98;
      letter-spacing: -0.04em;
    }}

    .hero p {{
      margin: 0;
      color: var(--muted);
      max-width: 980px;
      font-size: 15px;
      line-height: 1.35;
    }}

    .grid {{
      display: grid;
      grid-template-columns: 390px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }}

    .panel {{
      min-width: 0;
      padding: 22px;
    }}

    .section-title {{
      font-size: 13px;
      margin: 0 0 14px;
    }}

    .field-label {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: none;
      color: var(--muted);
      margin-bottom: 6px;
      font-weight: 650;
    }}

    input,
    select,
    button,
    textarea {{
      width: 100%;
      font: inherit;
      border-radius: 17px;
      border: 1px solid var(--line);
      padding: 11px 13px;
      background: rgba(255, 255, 255, 0.84);
      color: var(--ink);
      outline: none;
    }}

    input:focus,
    select:focus,
    textarea:focus {{
      border-color: rgba(11, 108, 88, 0.42);
      box-shadow: 0 0 0 3px rgba(11, 108, 88, 0.10);
    }}

    button {{
      cursor: pointer;
      font-weight: 750;
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
      color: white;
      border: 0;
      box-shadow: 0 14px 24px rgba(11, 108, 88, 0.20);
    }}

    button.secondary {{
      background: rgba(255, 255, 255, 0.74);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
    }}

    button:disabled {{
      cursor: not-allowed;
      opacity: 0.52;
    }}

    .stack > * + * {{
      margin-top: 12px;
    }}

    .examples {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }}

    .example-btn {{
      width: auto;
      padding: 10px 12px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.84);
      color: var(--ink);
      border: 1px solid var(--line);
      box-shadow: none;
      line-height: 1.1;
    }}

    .video-box {{
      border-radius: 24px;
      overflow: hidden;
      background: #101010;
      border: 1px solid var(--line);
      margin-bottom: 16px;
    }}

    video {{
      display: block;
      width: 100%;
      max-height: 430px;
      background: #101010;
    }}

    .cards {{
      display: flex;
      flex-wrap: wrap;
      justify-content: center;
      align-items: stretch;
      gap: 12px;
      max-width: 1090px;
      margin: 0 auto 16px;
    }}

    .card {{
      flex: 1 1 240px;
      max-width: 260px;
      min-width: 240px;
      border: 1px solid var(--line);
      border-radius: var(--radius-md);
      padding: 16px;
      min-height: 112px;
      background: var(--panel-strong);
      display: flex;
      flex-direction: column;
      justify-content: center;
    }}

    .card .k {{
      font-size: 12px;
      line-height: 1.12;
    }}

    .card .v {{
      font-size: 29px;
      font-weight: 800;
      margin-top: 12px;
      line-height: 1.03;
      letter-spacing: -0.04em;
    }}

    .viz-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
      margin-bottom: 16px;
      align-items: stretch;
    }}

    .detail-grid {{
      display: grid;
      grid-template-columns: minmax(0, 0.95fr) minmax(440px, 1.05fr);
      gap: 14px;
      align-items: stretch;
    }}

    .detail-card {{
      min-height: 280px;
    }}

    .viz-card {{
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel-strong);
      padding: 16px;
      overflow: hidden;
    }}

    .chart-toolbar {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 12px;
    }}

    .chart-toolbar .section-title {{
      margin: 0;
    }}

    .small-action {{
      width: auto;
      border-radius: 999px;
      padding: 8px 13px;
      font-size: 12px;
      letter-spacing: 0.04em;
      box-shadow: none;
      background: rgba(11, 108, 88, 0.10);
      color: var(--accent);
      border: 1px solid rgba(11, 108, 88, 0.22);
    }}

    .score-card {{
      padding: 18px;
    }}

    .timeline-shell {{
      min-height: 360px;
    }}

    .viz-card svg {{
      display: block;
      width: 100%;
      height: auto;
    }}

    .timeline-shell svg {{
      min-height: 340px;
    }}

    .empty-viz {{
      padding: 18px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }}

    .severity-votes-shell {{
      display: flex;
      align-items: center;
      min-height: 205px;
    }}

    .vote-list {{
      width: 100%;
      display: grid;
      gap: 14px;
    }}

    .vote-row {{
      display: grid;
      grid-template-columns: 96px minmax(120px, 1fr) 34px;
      gap: 14px;
      align-items: center;
    }}

    .vote-label {{
      color: var(--muted);
      font-size: 18px;
      line-height: 1.1;
    }}

    .vote-track {{
      height: 34px;
      border-radius: 999px;
      background: rgba(31, 26, 21, 0.06);
      overflow: hidden;
    }}

    .vote-bar {{
      height: 100%;
      min-width: 34px;
      border-radius: 999px;
      background: var(--accent);
    }}

    .vote-bar.major {{
      background: var(--danger);
    }}

    .vote-bar.moderate {{
      background: var(--warning);
    }}

    .vote-count {{
      font-size: 19px;
      font-weight: 750;
      font-variant-numeric: tabular-nums;
    }}

    .table-scroll {{
      width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      -webkit-overflow-scrolling: touch;
    }}

    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}

    th,
    td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}

    th {{
      font-size: 12px;
      line-height: 1.15;
    }}

    .top-window-list {{
      margin-top: 10px;
      border-top: 1px solid var(--line-strong);
      overflow: hidden;
    }}

    .top-window-row {{
      display: grid;
      grid-template-columns: 0.86fr 1.05fr 0.9fr 0.95fr;
      gap: 14px;
      align-items: center;
      padding: 9px 4px;
      border-bottom: 1px solid var(--line);
      min-width: 0;
    }}

    .top-window-header {{
      padding-top: 11px;
      padding-bottom: 8px;
    }}

    .mini-head {{
      font-size: 11px;
      letter-spacing: 0.08em;
      line-height: 1.1;
      white-space: nowrap;
    }}

    .mini-cell {{
      min-width: 0;
      font-size: 15px;
      line-height: 1.15;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .mono-ish {{
      font-variant-numeric: tabular-nums;
    }}

    .severity-pill {{
      display: inline-flex;
      align-items: center;
      justify-content: center;
      min-width: 58px;
      max-width: 100%;
      border-radius: 999px;
      padding: 5px 9px;
      font-size: 13px;
      font-weight: 750;
      background: rgba(11, 108, 88, 0.09);
      color: var(--accent);
      overflow: hidden;
      text-overflow: ellipsis;
    }}

    .severity-pill.major {{
      background: rgba(160, 60, 34, 0.10);
      color: var(--danger);
    }}

    .severity-pill.moderate {{
      background: rgba(193, 126, 23, 0.12);
      color: var(--warning);
    }}

    .links {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin: 0 0 18px;
    }}

    .links a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
      border-bottom: 1px solid rgba(11, 108, 88, 0.28);
    }}

    .muted {{
      color: var(--muted);
    }}

    .danger {{
      color: var(--danger);
    }}

    .json {{
      white-space: pre-wrap;
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
      background: rgba(31, 26, 21, 0.06);
      border-radius: 18px;
      padding: 16px;
      overflow: auto;
      max-height: 360px;
    }}


    .chart-modal {{
      position: fixed;
      inset: 0;
      z-index: 50;
      display: none;
      background: rgba(31, 26, 21, 0.42);
      backdrop-filter: blur(12px);
      padding: 28px;
    }}

    .chart-modal.active {{
      display: flex;
      align-items: center;
      justify-content: center;
    }}

    .chart-modal-card {{
      width: min(1480px, 96vw);
      max-height: 92vh;
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid var(--line);
      border-radius: 28px;
      box-shadow: 0 26px 80px rgba(31, 26, 21, 0.28);
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}

    .modal-head {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }}

    .modal-actions {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}

    .modal-actions button {{
      width: auto;
      padding: 8px 12px;
      border-radius: 999px;
      box-shadow: none;
      background: rgba(255, 255, 255, 0.86);
      color: var(--ink);
      border: 1px solid var(--line);
      font-size: 13px;
    }}

    .modal-chart-scroll {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 22px;
      background: rgba(255, 255, 255, 0.72);
      padding: 12px;
    }}

    .modal-chart-inner {{
      transform-origin: top left;
      transition: transform 120ms ease;
      width: 1320px;
      max-width: none;
    }}

    .modal-chart-inner svg {{
      width: 1320px;
      max-width: none;
      height: auto;
      min-height: 460px;
    }}

    @media (max-width: 1240px) {{
      .viz-grid,
      .detail-grid {{
        grid-template-columns: 1fr;
      }}
    }}

    @media (max-width: 980px) {{
      .wrap {{
        padding: 18px;
      }}

      .grid {{
        grid-template-columns: 1fr;
      }}

      .hero h1 {{
        font-size: 36px;
      }}
    }}

    @media (max-width: 620px) {{
      .top-window-row {{
        grid-template-columns: 0.8fr 1.1fr 0.9fr;
      }}

      .top-window-row > :nth-child(4) {{
        display: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="eyebrow">Demo interface</div>
      <h1>Upload a video for assessment and auto-annotation</h1>
      <p>The interface uses the project's saved models: it first scores accident probability across windows, then predicts severity for positive windows. The output includes a summary, window-level predictions, annotation segments, and an HTML/JSON bundle.</p>
    </section>

    <section class="grid">
      <section class="panel stack">
        <div>
          <div class="section-title">Configuration</div>
          <label class="field-label" for="accidentRun">Accident model directory</label>
          <input id="accidentRun" value="{accident_run_dir or ''}" />
        </div>

        <div>
          <label class="field-label" for="severityRun">Severity model directory</label>
          <input id="severityRun" value="{severity_run_dir or ''}" />
        </div>

        <div>
          <label class="field-label" for="device">Device</label>
          <select id="device">
            <option value="cuda">cuda</option>
            <option value="cpu">cpu</option>
          </select>
        </div>

        <div>
          <label class="field-label" for="threshold">Accident threshold</label>
          <input id="threshold" type="number" step="0.01" min="0" max="1" value="0.5" />
        </div>

        <div>
          <label class="field-label" for="windowSpan">Window size, frames</label>
          <input id="windowSpan" type="number" min="1" placeholder="auto" />
        </div>

        <div>
          <label class="field-label" for="stepFrames">Window step, frames</label>
          <input id="stepFrames" type="number" min="1" placeholder="auto" />
        </div>

        <div>
          <label class="field-label" for="maxWindows">Limit the number of windows</label>
          <input id="maxWindows" type="number" min="1" placeholder="no limit" />
        </div>

        <div>
          <label class="field-label" for="videoFile">Video</label>
          <input id="videoFile" type="file" accept="video/*" />
        </div>

        <button id="analyzeBtn" type="button">Run assessment</button>
        <button id="clearBtn" class="secondary" type="button">Clear result</button>

        <div>
          <div class="section-title">Quick examples</div>
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
          <div class="viz-card score-card">
            <div class="chart-toolbar">
              <div class="section-title">Score timeline</div>
              <button id="expandTimelineBtn" class="small-action" type="button" disabled>Expand chart</button>
            </div>
            <div id="timelineViz" class="timeline-shell empty-viz">The chart will appear after the assessment finishes.</div>
          </div>

          <div class="detail-grid">
            <div class="viz-card detail-card">
              <div class="section-title">Severity votes</div>
              <div id="severityVotesViz" class="severity-votes-shell empty-viz">The chart will appear after the assessment finishes.</div>
            </div>

            <div class="viz-card detail-card">
              <div class="section-title">Top windows</div>
              <div class="top-window-list" id="topWindowsBody">
                <div class="top-window-row top-window-header">
                  <div class="mini-head">Window</div>
                  <div class="mini-head">Time, sec</div>
                  <div class="mini-head">Score</div>
                  <div class="mini-head">Severity</div>
                </div>
                <div class="top-window-row"><div class="mini-cell muted">No data</div></div>
              </div>
            </div>
          </div>
        </div>

        <div class="links" id="artifactLinks"></div>

        <div class="section-title">Annotation segments</div>
        <div class="table-scroll">
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
        </div>

        <div class="section-title" style="margin-top:18px;">Window predictions</div>
        <div class="table-scroll">
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
        </div>

        <div class="section-title" style="margin-top:18px;">Raw JSON</div>
        <div id="jsonOutput" class="json"></div>
      </section>
    </section>
  </div>

  <div id="chartModal" class="chart-modal" aria-hidden="true">
    <div class="chart-modal-card">
      <div class="modal-head">
        <div class="section-title">Expanded score timeline</div>
        <div class="modal-actions">
          <button id="zoomOutBtn" type="button">−</button>
          <button id="zoomResetBtn" type="button">Reset</button>
          <button id="zoomInBtn" type="button">+</button>
          <button id="closeChartModalBtn" type="button">Close</button>
        </div>
      </div>
      <div class="modal-chart-scroll">
        <div id="modalChartInner" class="modal-chart-inner"></div>
      </div>
    </div>
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
    const expandTimelineBtn = document.getElementById("expandTimelineBtn");
    const chartModal = document.getElementById("chartModal");
    const modalChartInner = document.getElementById("modalChartInner");
    const closeChartModalBtn = document.getElementById("closeChartModalBtn");
    const zoomInBtn = document.getElementById("zoomInBtn");
    const zoomOutBtn = document.getElementById("zoomOutBtn");
    const zoomResetBtn = document.getElementById("zoomResetBtn");
    let timelineZoom = 1;

    function fmt(value, digits = 4) {{
      if (value === null || value === undefined || value === "") return "—";
      const num = Number(value);
      return Number.isFinite(num) ? num.toFixed(digits) : String(value);
    }}

    function safeNumber(value, fallback = 0) {{
      const num = Number(value);
      return Number.isFinite(num) ? num : fallback;
    }}

    function escapeHtml(value) {{
      return String(value ?? "—")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");
    }}

    function shortClipId(value) {{
      const text = String(value || "—");
      const match = text.match(/([0-9]{{3,5}})$/);
      if (match) return match[1];
      return text.length > 10 ? text.slice(-10) : text;
    }}

    function severityClass(value) {{
      const text = String(value || "").toLowerCase();
      if (text === "major") return "major";
      if (text === "moderate") return "moderate";
      return "minor";
    }}

    function applyTimelineZoom() {{
      modalChartInner.style.transform = `scale(${{timelineZoom}})`;
      modalChartInner.style.marginBottom = `${{Math.max(0, (timelineZoom - 1) * 460)}}px`;
      modalChartInner.style.marginRight = `${{Math.max(0, (timelineZoom - 1) * 1320)}}px`;
    }}

    function openTimelineModal() {{
      if (!timelineViz.querySelector("svg")) return;
      modalChartInner.innerHTML = timelineViz.innerHTML;
      timelineZoom = 1;
      applyTimelineZoom();
      chartModal.classList.add("active");
      chartModal.setAttribute("aria-hidden", "false");
    }}

    function closeTimelineModal() {{
      chartModal.classList.remove("active");
      chartModal.setAttribute("aria-hidden", "true");
      modalChartInner.innerHTML = "";
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
          <div class="k">${{escapeHtml(k)}}</div>
          <div class="v">${{escapeHtml(v)}}</div>
        </div>
      `).join("");
    }}

    function topWindows(windows, limit = 5) {{
      return [...windows]
        .sort((a, b) => safeNumber(b.accident_score) - safeNumber(a.accident_score))
        .slice(0, limit);
    }}

    function renderTopWindows(windows) {{
      const rows = topWindows(windows);

      const header = `
        <div class="top-window-row top-window-header">
          <div class="mini-head">Window</div>
          <div class="mini-head">Time, sec</div>
          <div class="mini-head">Score</div>
          <div class="mini-head">Severity</div>
        </div>
      `;

      if (!rows.length) {{
        topWindowsBody.innerHTML = header + `<div class="top-window-row"><div class="mini-cell muted">No data</div></div>`;
        return;
      }}

      topWindowsBody.innerHTML = header + rows.map((item) => {{
        const fullId = item.clip_id || "—";
        const shortId = shortClipId(fullId);
        const severity = item.severity_label || "—";
        const timeText = `${{fmt(item.start_sec, 1)}}–${{fmt(item.end_sec, 1)}}`;

        return `
          <div class="top-window-row">
            <div class="mini-cell mono-ish" title="${{escapeHtml(fullId)}}">${{escapeHtml(shortId)}}</div>
            <div class="mini-cell mono-ish">${{escapeHtml(timeText)}}</div>
            <div class="mini-cell mono-ish">${{fmt(item.accident_score, 3)}}</div>
            <div class="mini-cell"><span class="severity-pill ${{severityClass(severity)}}">${{escapeHtml(severity)}}</span></div>
          </div>
        `;
      }}).join("");
    }}

    function renderTimeline(payload) {{
      const windows = payload.window_predictions || [];

      if (!windows.length) {{
        timelineViz.className = "timeline-shell empty-viz";
        timelineViz.innerHTML = `<div>Not enough data to build a timeline.</div>`;
        expandTimelineBtn.disabled = true;
        return;
      }}

      const threshold = safeNumber(payload.summary.accident_threshold, 0.5);
      const duration = Math.max(
        safeNumber(payload.summary.duration_sec, 0),
        ...windows.map((item) => safeNumber(item.end_sec, 0)),
        1
      );

      const width = 1320;
      const height = 430;
      const chartLeft = 82;
      const chartTop = 34;
      const chartWidth = width - 140;
      const chartHeight = 270;
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
        const fill = score >= threshold ? "#a03c22" : "#0d6b55";

        linePoints.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
        areaPoints.push(`${{x.toFixed(2)}},${{y.toFixed(2)}}`);
        circles.push(`<circle cx="${{x.toFixed(2)}}" cy="${{y.toFixed(2)}}" r="5" fill="${{fill}}" stroke="white" stroke-width="2" />`);
      }});

      areaPoints.push(`${{chartLeft + chartWidth}},${{scoreBaseY}}`);

      const bandY = scoreBaseY + 44;
      const bands = (payload.segments || []).map((segment) => {{
        const startX = xPos(segment.start_sec);
        const endX = xPos(segment.end_sec);
        const bandWidth = Math.max(endX - startX, 4);
        const severity = segment.severity_label || "detected";
        const color = severity === "major" ? "#a03c22" : severity === "moderate" ? "#c17e17" : "#0d6b55";

        return `
          <rect x="${{startX.toFixed(2)}}" y="${{bandY.toFixed(2)}}" width="${{bandWidth.toFixed(2)}}" height="16" rx="8" fill="${{color}}" fill-opacity="0.92" />
          <text x="${{(startX + bandWidth / 2).toFixed(2)}}" y="${{(bandY + 12).toFixed(2)}}" text-anchor="middle" font-size="10" fill="white">${{escapeHtml(severity)}}</text>
        `;
      }}).join("");

      const thresholdY = yPos(threshold);

      timelineViz.className = "timeline-shell";
      timelineViz.innerHTML = `
        <svg viewBox="0 0 1320 430" role="img" aria-label="Score timeline by window">
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
          <text x="${{chartLeft}}" y="${{(bandY + 42).toFixed(2)}}" font-size="12" fill="#6f665c">Annotation segments</text>
        </svg>
      `;
      expandTimelineBtn.disabled = false;
    }}

    function renderSeverityVotes(payload) {{
      const votes = payload.severity_vote_counts || {{}};
      const labels = Object.entries(votes).sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));

      if (!labels.length) {{
        severityVotesViz.className = "severity-votes-shell empty-viz";
        severityVotesViz.innerHTML = `<div>No positive windows were found, so the severity chart is unavailable.</div>`;
        return;
      }}

      const maxValue = Math.max(...labels.map(([, value]) => Number(value)), 1);
      severityVotesViz.className = "severity-votes-shell";
      severityVotesViz.innerHTML = `
        <div class="vote-list">
          ${{labels.map(([label, value]) => {{
            const width = Math.max(8, (Number(value) / maxValue) * 100);
            return `
              <div class="vote-row">
                <div class="vote-label">${{escapeHtml(label)}}</div>
                <div class="vote-track"><div class="vote-bar ${{severityClass(label)}}" style="width:${{width.toFixed(1)}}%"></div></div>
                <div class="vote-count">${{escapeHtml(value)}}</div>
              </div>
            `;
          }}).join("")}}
        </div>
      `;
    }}

    function renderRows(payload) {{
      const segments = payload.segments || [];
      const windows = payload.window_predictions || [];

      segmentsBody.innerHTML = segments.length ? segments.map((item) => `
        <tr>
          <td>${{escapeHtml(item.segment_id)}}</td>
          <td>${{fmt(item.start_sec)}}</td>
          <td>${{fmt(item.end_sec)}}</td>
          <td>${{escapeHtml(item.severity_label || "—")}}</td>
          <td>${{fmt(item.max_accident_score)}}</td>
          <td>${{escapeHtml(item.window_count)}}</td>
        </tr>
      `).join("") : `<tr><td colspan="6">No segments detected</td></tr>`;

      windowsBody.innerHTML = windows.length ? windows.map((item) => `
        <tr>
          <td>${{escapeHtml(item.clip_id)}}</td>
          <td>${{fmt(item.start_sec)}}</td>
          <td>${{fmt(item.end_sec)}}</td>
          <td>${{fmt(item.accident_score)}}</td>
          <td>${{escapeHtml(item.accident_label || "—")}}</td>
          <td>${{escapeHtml(item.severity_label || "—")}}</td>
        </tr>
      `).join("") : `<tr><td colspan="6">No window predictions</td></tr>`;
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

      const response = await fetch("/api/analyze", {{
        method: "POST",
        body: formData,
      }});

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
      topWindowsBody.innerHTML = `
        <div class="top-window-row top-window-header">
          <div class="mini-head">Window</div>
          <div class="mini-head">Time, sec</div>
          <div class="mini-head">Score</div>
          <div class="mini-head">Severity</div>
        </div>
        <div class="top-window-row"><div class="mini-cell muted">No data</div></div>
      `;
      timelineViz.className = "timeline-shell empty-viz";
      timelineViz.innerHTML = `<div>The chart will appear after the assessment finishes.</div>`;
      severityVotesViz.className = "severity-votes-shell empty-viz";
      severityVotesViz.innerHTML = `<div>The chart will appear after the assessment finishes.</div>`;
      expandTimelineBtn.disabled = true;
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


    expandTimelineBtn.addEventListener("click", openTimelineModal);
    closeChartModalBtn.addEventListener("click", closeTimelineModal);
    chartModal.addEventListener("click", (event) => {{
      if (event.target === chartModal) closeTimelineModal();
    }});
    zoomInBtn.addEventListener("click", () => {{
      timelineZoom = Math.min(2.2, timelineZoom + 0.2);
      applyTimelineZoom();
    }});
    zoomOutBtn.addEventListener("click", () => {{
      timelineZoom = Math.max(0.7, timelineZoom - 0.2);
      applyTimelineZoom();
    }});
    zoomResetBtn.addEventListener("click", () => {{
      timelineZoom = 1;
      applyTimelineZoom();
    }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === "Escape" && chartModal.classList.contains("active")) closeTimelineModal();
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

    def do_GET(self) -> None:
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

    def do_POST(self) -> None:
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

        except Exception as exc:
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