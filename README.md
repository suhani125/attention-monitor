# Smart Attention Monitoring System

Real-time AI-based attention monitoring system that analyzes eye movements
and head orientation to detect drowsiness and distraction — for one person
or a whole room, using computer vision (OpenCV + MediaPipe) instead of a
physical sensor.

## Features

- **Real-time facial landmark detection** via MediaPipe's FaceLandmarker
  (Tasks API), the current, actively-maintained MediaPipe interface.
- **Eye closure detection** via the Eye Aspect Ratio (EAR) algorithm.
- **3D head-pose estimation** (pitch/yaw/roll) using OpenCV's `solvePnP`
  against a generic 3D face model — a proper geometric pose solve rather
  than a flat 2D heuristic, so "looking away" detection accounts for both
  turning the head (yaw) and tilting it up/down (pitch).
- **Multi-person tracking**: track several faces at once (`--max-faces N`),
  each with its own persistent ID, attention score, and independent
  drowsy/distracted state — built for classroom-style monitoring, not just
  a single driver/user.
- **Per-person auto-calibration**: the first few seconds each face is seen,
  the system just measures *that person's* normal eyes-open EAR and derives
  a personalized closed-eye threshold from it — no more one-size-fits-all
  threshold across different eye shapes/camera angles.
- **Live web dashboard** (optional, `--dashboard`): a small Flask page,
  viewable from any device on the network, showing every tracked person's
  current state and a live attention-score graph, auto-refreshing every
  second. Fully offline — the charting library is bundled locally
  (`static/chart.umd.min.js`), so it works even with no internet access or
  behind a restrictive firewall.
- **Automated behavioral logging**: every drowsiness alert and distraction
  episode, per person, is timestamped and written to
  `logs/attention_log.csv`.
- **Real-time visual feedback and alerts**: per-person bounding boxes and
  eye landmarks, color-coded state labels, a full-frame red flash +
  optional audible beep on drowsiness.
- **Low-latency pipeline**: single-pass per-frame processing, no GPU
  required.

## How it works

1. Each frame is passed through MediaPipe FaceLandmarker to extract 478
   facial landmarks per detected face.
2. Faces are matched frame-to-frame by a lightweight nearest-centroid
   tracker, so each person keeps a stable ID as long as they stay roughly in
   place (or are the closest match after moving).
3. For the first `--calibration-seconds` (default 4) a person is visible,
   the system records their EAR while assuming eyes are mostly open, then
   sets that person's closed-eye threshold to `calibration_factor × median
   baseline EAR` (default factor 0.75).
4. Six landmarks (nose tip, chin, eye corners, mouth corners) are matched
   against a generic 3D face model via `cv2.solvePnP` to recover head
   orientation (pitch/yaw/roll) in degrees.
5. If EAR stays below that person's threshold for N consecutive frames
   (default 15, ~0.5s at 30fps), a drowsiness alert fires. If yaw or pitch
   exceeds their thresholds, the person is marked distracted.
6. Both signals feed a rolling per-person attention score; every state
   transition is written to the CSV log and (if enabled) pushed to the live
   dashboard.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9–3.13. On first run, `main.py` automatically downloads
the required FaceLandmarker model (~4 MB) to `models/face_landmarker.task`
— this needs a normal internet connection the first time only.

## Usage

```bash
# Single person, default webcam
python main.py

# Classroom mode: track up to 6 faces at once
python main.py --max-faces 6

# Also serve a live dashboard at http://localhost:5000
python main.py --max-faces 6 --dashboard

# Custom dashboard port
python main.py --dashboard --dashboard-port 8080

# Run on a video file instead of a live camera
python main.py --source path/to/video.mp4

# Disable the audible beep
python main.py --no-sound

# Tune sensitivity
python main.py --consec-frames 20 --yaw-threshold 30 --pitch-threshold 25

# Shorter/longer per-person calibration window
python main.py --calibration-seconds 6
```

**Controls while the OpenCV window is focused:**
| Key | Action |
|-----|--------|
| `q` / `Esc` | Quit |
| `s` | Print current session stats (per-person blinks, drowsy events, distraction events) to the console |

**Dashboard:** with `--dashboard`, open `http://localhost:5000` (or your
chosen port) in a browser on the same machine, or `http://<your-ip>:5000`
from another device on the same network (e.g. a teacher's laptop watching a
classroom webcam feed running on a different computer).

## Project structure

```
attention-monitor/
├── attention_engine.py   # Core detection: landmarks, EAR, 3D head pose,
│                         # multi-person tracking, calibration, scoring, logging
├── main.py                # Webcam/video app: overlays, alerts, CLI, dashboard wiring
├── dashboard.py            # Flask live dashboard (state store + HTML/JS page)
├── static/
│   └── chart.umd.min.js    # Chart.js bundled locally (dashboard works fully offline)
├── requirements.txt
├── models/
│   └── face_landmarker.task  # Auto-downloaded on first run
├── logs/
│   └── attention_log.csv     # Auto-created on first run
└── README.md
```

## CLI reference

| Flag | Default | Meaning |
|---|---|---|
| `--source` | `0` | Camera index or path to a video file |
| `--max-faces` | `1` | Max number of people to track simultaneously |
| `--calibration-seconds` | `4.0` | Per-person auto-calibration window |
| `--ear-threshold` | `0.21` | Fallback EAR threshold if calibration doesn't gather enough samples |
| `--consec-frames` | `15` | Consecutive closed-eye frames to trigger a drowsiness alert |
| `--yaw-threshold` | `25.0` | Head yaw (°) considered "looking away" |
| `--pitch-threshold` | `20.0` | Head pitch (°) considered "looking away" |
| `--dashboard` | off | Enable the live web dashboard |
| `--dashboard-port` | `5000` | Dashboard port |
| `--no-sound` | off | Disable the audible drowsiness beep |

## Extending this project

- Add yawn detection using the mouth-aspect-ratio (MAR), the same way EAR is
  computed for eyes.
- Export end-of-session summaries (per-person average attention, total
  drowsy episodes) as a PDF report.
- Swap the polling dashboard for WebSockets if you want sub-second updates
  across many simultaneous viewers.
- Persist calibration per known person across sessions (e.g. keyed by a
  face-recognition embedding) instead of recalibrating every time someone
  reappears.
