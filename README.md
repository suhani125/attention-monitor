# Smart Attention Monitoring System

I built this to see if a normal webcam could tell when someone is getting sleepy or looking away. It turns out it can, and it works for one person or a whole classroom. No special hardware, just OpenCV and MediaPipe.

## What it does

It finds faces in the video and watches two things: whether the eyes stay closed for too long (drowsy), and whether the head is turned away from the screen (distracted). Each person gets their own ID and attention score, so you can track several people at the same time.

Some other things I added along the way:

- It calibrates for each person in the first few seconds, because everyone's eyes are a bit different.
- There's an optional web dashboard so you can watch from another laptop. It works offline too.
- Every drowsy or distracted moment gets saved to `logs/attention_log.csv` with a timestamp.
- When someone is drowsy, the screen flashes red and it can beep.
- It runs fine on a CPU, no GPU needed.

## Install

```bash
pip install -r requirements.txt
```

You need Python 3.9 to 3.13. The first time you run it, it downloads a small model (about 4 MB), so you need internet just once.

## Run it

```bash
python main.py                              # one person, default webcam
python main.py --max-faces 6                # classroom mode
python main.py --max-faces 6 --dashboard    # with the live dashboard
python main.py --source video.mp4           # use a video file instead
python main.py --no-sound                   # turn off the beep
```

Press `q` or `Esc` to quit, and `s` to print the session stats.

For the dashboard, open `http://localhost:5000`. From another device on the same network, use `http://<your-ip>:5000`.

## Options

| Option | Default | What it does |
|---|---|---|
| `--source` | 0 | Camera number or video file |
| `--max-faces` | 1 | How many people to track |
| `--calibration-seconds` | 4 | How long it spends learning each person's normal eyes |
| `--ear-threshold` | 0.21 | Backup value if calibration doesn't get enough data |
| `--consec-frames` | 15 | How many closed-eye frames before a drowsy alert |
| `--yaw-threshold` | 25 | Left/right head turn (degrees) that counts as looking away |
| `--pitch-threshold` | 20 | Up/down head tilt (degrees) that counts as looking away |
| `--dashboard` | off | Turn on the dashboard |
| `--dashboard-port` | 5000 | Dashboard port |
| `--no-sound` | off | Turn off the beep |

If it feels too sensitive or not sensitive enough, try changing `--consec-frames` and the two threshold values first.

## Files

- `attention_engine.py` has the detection, tracking and scoring
- `main.py` is the app itself (overlays, alerts, command line options)
- `dashboard.py` is the Flask dashboard
- `static/` has the chart library, saved locally so the dashboard works offline
- `models/` and `logs/` get created automatically

## Things I want to add

- Yawn detection
- PDF summary reports
- A faster dashboard using WebSockets
- Remembering calibration for people it has seen before
