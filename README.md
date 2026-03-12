# Anti Doomscroll

`anti_doomscroll.py` watches your webcam for a sustained phone-looking pattern and interrupts you with a fullscreen VLC video when it thinks you are doom-scrolling.

## What it does

- tracks face landmarks with MediaPipe
- estimates downward gaze with iris position plus head pitch
- optionally confirms a phone is visible before triggering
- looks for scrolling-like motion inside the detected phone region
- fires a fullscreen VLC clip after the pattern holds for long enough

## Requirements

- Windows machine with a webcam
- Python 3.11+ (tested here with Python 3.12)
- VLC installed if you want video playback instead of text-only warnings

Install Python packages:

```bash
pip install -r requirements.txt
```

## Model files

Place these files next to `anti_doomscroll.py`:

- required for MediaPipe Tasks fallback: `face_landmarker.task`
- optional phone detector: `efficientdet_lite2.tflite`

Download links:

- face landmarker: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task`
- phone detector: `https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite2/float32/latest/efficientdet_lite2.tflite`

If the phone detector is missing, the script can still run in gaze-only mode with `--disable-phone-check`.

## Run

```bash
python anti_doomscroll.py
```

Useful options:

```bash
python anti_doomscroll.py --sensitivity high
python anti_doomscroll.py --delay 3 --rearm-seconds 2
python anti_doomscroll.py --disable-phone-check
python anti_doomscroll.py --camera 1
python anti_doomscroll.py --video annoying.mp4
```

## How the trigger works

In full mode, the script waits for all of these signals:

1. your gaze trends downward
2. a phone is detected in the lower part of the frame
3. your face orientation points toward that phone
4. the phone region shows repeated motion that looks like scrolling
5. the pattern lasts for the configured delay

If phone detection is disabled, the trigger falls back to gaze only.

## Tips

- keep the debug window visible while tuning thresholds
- use `--calibration-seconds 1.5` or higher for a better neutral baseline
- lower sensitivity means fewer false positives; higher sensitivity reacts sooner
- if VLC is not installed, the script still shows on-screen warnings in the debug feed

## Troubleshooting

- `Could not open camera index`: try `--camera 1` or another device index
- `Missing required packages`: reinstall with `pip install -r requirements.txt`
- `Missing face landmarker model`: download `face_landmarker.task` and place it beside the script
- no trigger in phone mode: lower `--phone-score`, improve lighting, or try `--disable-phone-check`
