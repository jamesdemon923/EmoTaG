"""Face-centered square crop -> 512x512 @ 25 FPS.

Prepares a raw talking-head video for the EmoTaG pipeline: detects the face over
a sample of frames, takes a square region centered on the median face box
(side = `--expand` x the face size, clamped to the frame), and writes the crop
resized to 512x512 at 25 FPS with the audio track preserved. HDTF clips are
already distributed with official crop metadata; use this for MEAD / custom
videos.

Usage:
    python tools/crop_video.py --input raw.mp4 --output data/<ID>/<ID>.mp4
    # inspect the chosen crop without writing the video:
    python tools/crop_video.py --input raw.mp4 --output out.mp4 --dry_run
"""
import argparse
import os
import subprocess

import cv2
import numpy as np


def detect_face_box(video, num_samples=25):
    """Median face box (x, y, w, h) over sampled frames, via OpenCV's detector."""
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    boxes = []
    for idx in np.linspace(0, max(n - 1, 0), num_samples).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(found):
            boxes.append(max(found, key=lambda b: b[2] * b[3]))  # largest face
    cap.release()
    if not boxes:
        raise SystemExit('No face detected; crop the video manually.')
    return np.median(np.stack(boxes), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='raw source video')
    ap.add_argument('--output', required=True, help='cropped 512x512@25fps video')
    ap.add_argument('--size', type=int, default=512)
    ap.add_argument('--fps', type=int, default=25)
    ap.add_argument('--expand', type=float, default=2.2,
                    help='crop side = expand x face size (2.0-2.5 keeps head+shoulders)')
    ap.add_argument('--dry_run', action='store_true', help='print the crop, do not encode')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.input)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    x, y, w, h = detect_face_box(args.input)
    side = int(min(args.expand * max(w, h), min(W, H)))   # square, inside the frame
    cx, cy = x + w / 2, y + h / 2 - 0.1 * side            # bias up: keep hair, show shoulders
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))
    print(f'source {W}x{H}  face ({int(w)}x{int(h)})  ->  crop {side}x{side}+{x0}+{y0}')
    if args.dry_run:
        return

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    subprocess.check_call([
        'ffmpeg', '-y', '-loglevel', 'error', '-i', args.input,
        '-vf', f'crop={side}:{side}:{x0}:{y0},scale={args.size}:{args.size}',
        '-r', str(args.fps), '-c:v', 'libx264', '-crf', '15', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', args.output,
    ])
    print('wrote', args.output)


if __name__ == '__main__':
    main()
