"""
dashboard.py
------------
Optional live web dashboard for the Smart Attention Monitoring System.

Runs a small Flask app in a background thread so it doesn't block the main
webcam loop. The main app pushes per-person snapshots into a thread-safe
`DashboardState` object (via the AttentionEngine's `on_person_update`
callback); the dashboard polls that state and serves it as JSON, and the
browser page polls the JSON endpoint to redraw a live chart.

This is intentionally simple (polling, not websockets) so it has zero extra
dependencies beyond Flask and works reliably on any network without extra
setup.
"""

from __future__ import annotations

import threading
from collections import deque
from pathlib import Path
from typing import Optional

from flask import Flask, jsonify, render_template_string

STATIC_DIR = Path(__file__).parent / "static"

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Attention Monitor Dashboard</title>
  <script src="/static/chart.umd.min.js"></script>
  <style>
    body { font-family: Arial, Helvetica, sans-serif; background: #ffffff; color: #222222; margin: 20px; }
    h2 { font-size: 18px; margin-bottom: 4px; }
    p.note { color: #666666; font-size: 13px; margin-top: 0; margin-bottom: 16px; }
    table { border-collapse: collapse; width: 100%; max-width: 800px; font-size: 14px; margin-bottom: 24px; }
    th, td { border: 1px solid #cccccc; padding: 6px 10px; text-align: left; }
    th { background: #f0f0f0; }
    .state-FOCUSED { color: #2e7d32; font-weight: bold; }
    .state-DROWSY { color: #c62828; font-weight: bold; }
    .state-DISTRACTED { color: #e65100; font-weight: bold; }
    .state-CALIBRATING { color: #777777; font-weight: bold; }
    .empty-msg { color: #888888; padding: 10px; }
    #chart-box { max-width: 800px; border: 1px solid #cccccc; padding: 10px; }
    canvas { max-height: 300px; }
  </style>
</head>
<body>
  <h2>Attention Monitor - Live Dashboard</h2>
  <p class="note">Refreshes every second. One row per person detected on camera.</p>

  <table id="table">
    <thead>
      <tr><th>Person</th><th>State</th><th>Attention</th><th>EAR</th><th>Pitch</th><th>Yaw</th><th>Roll</th></tr>
    </thead>
    <tbody id="table-body">
      <tr><td colspan="7" class="empty-msg">Waiting for a face to appear on camera...</td></tr>
    </tbody>
  </table>

  <div id="chart-box">
    <canvas id="chart"></canvas>
  </div>

  <script>
    let chart = null;
    try {
      if (typeof Chart !== 'undefined') {
        const ctx = document.getElementById('chart').getContext('2d');
        chart = new Chart(ctx, {
          type: 'line',
          data: { datasets: [] },
          options: {
            responsive: true,
            animation: false,
            scales: {
              x: { type: 'linear', title: { display: true, text: 'seconds ago' } },
              y: { min: 0, max: 100, title: { display: true, text: 'attention score' } }
            }
          }
        });
      } else {
        console.warn('Chart.js failed to load from /static/chart.umd.min.js. Table will still update.');
        const canvas = document.getElementById('chart');
        canvas.replaceWith(document.createTextNode(
          'Graph unavailable (chart library failed to load). The table above still updates live.'
        ));
      }
    } catch (e) {
      console.error('Chart init failed:', e);
      chart = null;
    }

    const colors = ['#1565c0', '#c62828', '#2e7d32', '#e65100', '#6a1b9a', '#00838f'];

    async function refresh() {
      let data;
      try {
        const res = await fetch('/api/state');
        data = await res.json();
      } catch (e) {
        console.error('Failed to fetch /api/state:', e);
        return;
      }

      const ids = Object.keys(data);
      const tbody = document.getElementById('table-body');
      if (ids.length === 0) {
        tbody.innerHTML = '<tr><td colspan="7" class="empty-msg">Waiting for a face to appear on camera...</td></tr>';
      } else {
        tbody.innerHTML = ids.map(id => {
          const p = data[id];
          return `<tr>
            <td>#${id}</td>
            <td class="state-${p.state}">${p.state}</td>
            <td>${p.state === 'CALIBRATING' ? '-' : (p.score ?? 0).toFixed(0) + '%'}</td>
            <td>${p.ear != null ? p.ear.toFixed(2) : '-'}</td>
            <td>${p.pitch != null ? p.pitch.toFixed(0) + ' deg' : '-'}</td>
            <td>${p.yaw != null ? p.yaw.toFixed(0) + ' deg' : '-'}</td>
            <td>${p.roll != null ? p.roll.toFixed(0) + ' deg' : '-'}</td>
          </tr>`;
        }).join('');
      }

      if (chart) {
        try {
          const now = Date.now() / 1000;
          chart.data.datasets = ids.map((id, i) => {
            const hist = data[id].history || [];
            return {
              label: 'Person #' + id,
              borderColor: colors[i % colors.length],
              backgroundColor: 'transparent',
              borderWidth: 2,
              pointRadius: 0,
              data: hist.map(([ts, score]) => ({ x: -(now - ts), y: score })),
            };
          });
          chart.update();
        } catch (e) {
          console.error('Chart update failed:', e);
        }
      }
    }

    setInterval(refresh, 1000);
    refresh();
  </script>
</body>
</html>
"""


class DashboardState:
    """Thread-safe store of the latest per-person attention snapshots plus a
    bounded score history for graphing. Written to by the main video loop
    (via AttentionEngine's on_person_update callback), read by Flask
    request handlers running on a different thread."""

    def __init__(self, history_len: int = 300):
        self._lock = threading.Lock()
        self._history_len = history_len
        self._persons: dict[int, dict] = {}

    def update_person(self, person_id: int, snapshot: dict) -> None:
        with self._lock:
            entry = self._persons.setdefault(
                person_id, {"history": deque(maxlen=self._history_len)}
            )
            entry.update(snapshot)
            entry["history"].append((snapshot["timestamp"], snapshot["score"]))

    def prune(self, active_ids: set[int]) -> None:
        with self._lock:
            for pid in list(self._persons.keys()):
                if pid not in active_ids:
                    del self._persons[pid]

    def snapshot(self) -> dict:
        with self._lock:
            return {
                str(pid): {
                    "state": p.get("state"),
                    "score": p.get("score"),
                    "ear": p.get("ear"),
                    "pitch": p.get("pitch"),
                    "yaw": p.get("yaw"),
                    "roll": p.get("roll"),
                    "history": list(p["history"]),
                }
                for pid, p in self._persons.items()
            }


def create_dashboard_app(state: DashboardState) -> Flask:
    app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="/static")
    app.logger.disabled = True
    import logging
    logging.getLogger("werkzeug").setLevel(logging.ERROR)

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/state")
    def api_state():
        return jsonify(state.snapshot())

    return app


def start_dashboard_thread(
    state: DashboardState, host: str = "0.0.0.0", port: int = 5000
) -> threading.Thread:
    """Start the Flask dashboard in a background daemon thread and return it."""
    app = create_dashboard_app(state)

    def _run():
        app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread