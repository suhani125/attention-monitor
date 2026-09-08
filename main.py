"""
main.py
-------
Real-time Smart Attention Monitoring System.

Run:
    python main.py                          # default webcam, single person
    python main.py --max-faces 4             # classroom mode, up to 4 people
    python main.py --source video.mp4        # run on a video file instead
    python main.py --dashboard               # also serve a live web dashboard
    python main.py --dashboard --dashboard-port 8080
    python main.py --no-sound                # disable audible alert beep

Controls while the OpenCV window is focused:
    q or ESC  -> quit
    s         -> print current session stats to console

What's new in this version
---------------------------
- **3D head pose** (pitch/yaw/roll via OpenCV solvePnP) replaces the old 2D
  nose-offset heuristic for "looking away" detection.
- **Multi-person tracking**: each face on screen gets its own persistent ID,
  attention score, and drowsiness/distraction state — useful for a
  classroom, not just a single driver/user.
- **Per-person auto-calibration**: for the first few seconds a face is seen,
  the system just measures that person's normal (eyes-open) EAR baseline
  and derives a personalized closed-eye threshold from it, instead of using
  one fixed threshold for everyone.
- **Live web dashboard** (optional, `--dashboard`): a Flask page other
  devices on the network can open to see everyone's attention score update
  in real time (e.g. a teacher's laptop watching a classroom webcam).
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np

from attention_engine import AttentionEngine, EngineConfig, LEFT_EYE, RIGHT_EYE

STATE_COLORS = {
    "FOCUSED": (60, 200, 60),      # green
    "DROWSY": (0, 0, 255),         # red
    "DISTRACTED": (0, 165, 255),   # orange
    "CALIBRATING": (180, 180, 180),  # gray
}

PERSON_PALETTE = [
    (60, 200, 60), (255, 184, 90), (200, 120, 255), (90, 200, 255),
    (255, 120, 120), (120, 255, 220),
]


def beep():
    """Best-effort terminal beep; safe no-op if unsupported."""
    try:
        sys.stdout.write("\a")
        sys.stdout.flush()
    except Exception:
        pass


def draw_eye_landmarks(frame, pts, indices, color):
    for i in indices:
        x, y = int(pts[i][0]), int(pts[i][1])
        cv2.circle(frame, (x, y), 2, color, -1)


def face_bbox(pts, pad=20):
    x_min, y_min = pts[:, 0].min(), pts[:, 1].min()
    x_max, y_max = pts[:, 0].max(), pts[:, 1].max()
    return (int(x_min - pad), int(y_min - pad), int(x_max + pad), int(y_max + pad))


def draw_person_overlay(frame, result, color):
    pts = result.landmarks
    x1, y1, x2, y2 = face_bbox(pts)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    draw_eye_landmarks(frame, pts, LEFT_EYE, color)
    draw_eye_landmarks(frame, pts, RIGHT_EYE, color)

    if result.state == "CALIBRATING":
        pct = int(result.calibration_progress * 100)
        label = f"#{result.person_id} calibrating {pct}%"
    else:
        label = f"#{result.person_id} {result.state} {result.attention_score:.0f}%"

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (0, 0, 0), 2, cv2.LINE_AA)


def run(source, sound_enabled: bool, args):
    dashboard_state = None
    if args.dashboard:
        from dashboard import DashboardState, start_dashboard_thread
        dashboard_state = DashboardState()
        start_dashboard_thread(dashboard_state, host="0.0.0.0", port=args.dashboard_port)
        print(f"[dashboard] Live dashboard running at http://localhost:{args.dashboard_port}")

    def on_person_update(person_id, snapshot):
        if dashboard_state is not None:
            dashboard_state.update_person(person_id, snapshot)

    cfg = EngineConfig(
        ear_threshold_default=args.ear_threshold,
        ear_consec_frames=args.consec_frames,
        yaw_threshold_deg=args.yaw_threshold,
        pitch_threshold_deg=args.pitch_threshold,
        max_faces=args.max_faces,
        calibration_seconds=args.calibration_seconds,
    )
    engine = AttentionEngine(cfg, on_person_update=on_person_update)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, int):
        # Common Windows fix: the default backend sometimes fails silently
        # on Windows webcams. Retry explicitly with DirectShow.
        print("[info] Default backend failed to open camera, retrying with DirectShow...")
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {source}")
        print("  - Is another app (Zoom/Teams/browser) using the camera?")
        print("  - Try a different index: python main.py --source 1")
        print("  - Check Windows Settings > Privacy & security > Camera")
        return

    # Sanity check: make sure we can actually read a frame before entering
    # the main loop, so failures show up as a clear message instead of a
    # silently blank/frozen window.
    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        print("[ERROR] Camera opened but returned no frame data.")
        print("  - Another app may have an exclusive lock on the camera.")
        print("  - Try unplugging/replugging a USB webcam, or reboot if this persists.")
        cap.release()
        return

    prev_time = time.time()
    fps_smooth = 0.0
    last_beep_time = 0.0

    print("Smart Attention Monitoring System running. Press 'q' to quit, 's' for stats.")
    if args.max_faces > 1:
        print(f"Multi-person mode: tracking up to {args.max_faces} faces.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of stream or camera read failed.")
            break

        results = engine.process_frame(frame)

        if dashboard_state is not None:
            dashboard_state.prune(set(engine.active_person_ids()))

        if not results:
            cv2.putText(frame, "No face detected", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (150, 150, 150), 2, cv2.LINE_AA)
        else:
            any_drowsy = False
            any_distracted = False
            any_calibrating = False

            for i, result in enumerate(results):
                color = STATE_COLORS.get(result.state, (255, 255, 255))
                if len(results) > 1:
                    color = PERSON_PALETTE[result.person_id % len(PERSON_PALETTE)]
                draw_person_overlay(frame, result, color)

                if result.state == "DROWSY":
                    any_drowsy = True
                elif result.state == "DISTRACTED":
                    any_distracted = True
                elif result.state == "CALIBRATING":
                    any_calibrating = True

            if any_calibrating and len(results) == 1:
                cv2.putText(frame, "Calibrating... look at the camera with eyes open",
                            (15, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (200, 200, 200), 2, cv2.LINE_AA)

            if any_drowsy:
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 8)
                cv2.putText(frame, "!!! DROWSINESS ALERT !!!", (15, frame.shape[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3, cv2.LINE_AA)
                if sound_enabled and (time.time() - last_beep_time) > 1.5:
                    beep()
                    last_beep_time = time.time()
            elif any_distracted:
                cv2.putText(frame, "Attention drifting - looking away", (15, frame.shape[0] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

        # --- FPS ---
        now = time.time()
        fps = 1.0 / max(now - prev_time, 1e-6)
        fps_smooth = fps if fps_smooth == 0 else (0.9 * fps_smooth + 0.1 * fps)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (frame.shape[1] - 140, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        cv2.imshow("Smart Attention Monitoring System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        elif key == ord("s"):
            print("Session stats:", engine.session_stats())

    print("Final session stats:", engine.session_stats())
    cap.release()
    cv2.destroyAllWindows()
    engine.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Smart Attention Monitoring System")
    parser.add_argument(
        "--source", default="0",
        help="Camera index (e.g. 0, 1) or path to a video file. Default: 0",
    )
    parser.add_argument("--no-sound", action="store_true", help="Disable audible alerts")
    parser.add_argument(
        "--ear-threshold", type=float, default=0.21,
        help="Fallback Eye Aspect Ratio threshold if per-person calibration doesn't "
             "gather enough samples (default 0.21). Calibration normally overrides this.",
    )
    parser.add_argument(
        "--consec-frames", type=int, default=15,
        help="Consecutive closed-eye frames required to trigger a drowsiness alert (default 15)",
    )
    parser.add_argument(
        "--yaw-threshold", type=float, default=25.0,
        help="Head yaw (left/right turn), in degrees, considered 'looking away' (default 25)",
    )
    parser.add_argument(
        "--pitch-threshold", type=float, default=20.0,
        help="Head pitch (up/down tilt), in degrees, considered 'looking away' (default 20)",
    )
    parser.add_argument(
        "--calibration-seconds", type=float, default=4.0,
        help="Seconds spent auto-calibrating each new person's EAR baseline (default 4.0)",
    )
    parser.add_argument(
        "--max-faces", type=int, default=1,
        help="Maximum number of faces to track simultaneously (default 1). "
             "Set higher (e.g. 4-10) for classroom/multi-person mode.",
    )
    parser.add_argument(
        "--dashboard", action="store_true",
        help="Serve a live web dashboard (Flask) showing everyone's attention in real time.",
    )
    parser.add_argument(
        "--dashboard-port", type=int, default=5000,
        help="Port for the live web dashboard (default 5000).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    source = int(args.source) if args.source.isdigit() else args.source
    run(source=source, sound_enabled=not args.no_sound, args=args)
