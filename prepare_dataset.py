"""
prepare_dataset.py
──────────────────
Extracts optical-flow feature sequences from UCF-Crime (or any labeled video
folder) and saves a ready-to-train .npz file.

Folder structure expected:
    data/
        panic/       ← Riot, Explosion, RoadAccidents, Assault videos
        normal/      ← Normal_Videos_for_Testing videos

Run:
    python prepare_dataset.py --data_dir ./data --out dataset.npz
"""

import os
import argparse
import numpy as np
import cv2
from tqdm import tqdm


# ── Feature extraction (same signals as your existing app) ──────────────────

def compute_direction_entropy(flow):
    angles = np.arctan2(flow[..., 1], flow[..., 0])
    hist, _ = np.histogram(angles, bins=8, range=(-np.pi, np.pi))
    hist = hist / (hist.sum() + 1e-10)
    return float(-np.sum(hist * np.log(hist + 1e-10)) / np.log(8))


def compute_crowd_density(frame, bg_sub):
    fg = bg_sub.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)
    return float(np.count_nonzero(fg) / fg.size)


def compute_flow_divergence(flow):
    fx, fy = flow[..., 0], flow[..., 1]
    return float(np.mean(np.abs(np.gradient(fx, axis=1) + np.gradient(fy, axis=0))) / 10.0)


def compute_turbulence(magnitude):
    return float(np.var(magnitude))


def extract_features_from_video(video_path, window=30, sample_fps=6):
    """
    Returns a list of feature windows, each shaped (window, 6).
    Features: [avg_speed, entropy, density, acceleration, turbulence, divergence]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    interval = max(1, int(fps / sample_fps))

    bg_sub = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=40)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return []

    prev = cv2.resize(prev, (320, 180))
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    frame_num = 0
    prev_speed = 0.0
    frame_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        if frame_num % interval != 0:
            continue

        frame = cv2.resize(frame, (320, 180))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_speed = float(np.mean(magnitude))
        entropy   = compute_direction_entropy(flow)
        density   = compute_crowd_density(frame, bg_sub)
        turb      = compute_turbulence(magnitude)
        div       = compute_flow_divergence(flow)
        accel     = abs(avg_speed - prev_speed)
        prev_speed = avg_speed

        frame_features.append([avg_speed, entropy, density, accel, turb, div])
        prev_gray = gray

    cap.release()

    # Sliding windows of length `window`
    windows = []
    for i in range(len(frame_features) - window + 1):
        windows.append(frame_features[i:i + window])

    return windows


# ── Main ─────────────────────────────────────────────────────────────────────

def build_dataset(data_dir, window=30, sample_fps=6):
    X, y = [], []

    for label_name, label_val in [("panic", 1), ("normal", 0)]:
        folder = os.path.join(data_dir, label_name)
        if not os.path.isdir(folder):
            print(f"[WARN] Folder not found: {folder}")
            continue

        videos = [f for f in os.listdir(folder) if f.lower().endswith(
            ('.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv')
        )]
        print(f"\nProcessing {len(videos)} '{label_name}' videos...")

        for vid in tqdm(videos):
            path = os.path.join(folder, vid)
            windows = extract_features_from_video(path, window=window, sample_fps=sample_fps)
            for w in windows:
                X.append(w)
                y.append(label_val)

    X = np.array(X, dtype=np.float32)   # (N, window, 6)
    y = np.array(y, dtype=np.float32)   # (N,)

    # Also produce a continuous panic score (0–100) for regression
    # For binary labels: normal=10, panic=85 as soft targets
    y_score = np.where(y == 1, 85.0, 10.0).astype(np.float32)

    print(f"\nDataset: {X.shape[0]} samples  |  panic: {int(y.sum())}  |  normal: {int((1-y).sum())}")
    return X, y, y_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   default="./data",       help="Root folder with panic/ and normal/ subfolders")
    parser.add_argument("--out",        default="dataset.npz",  help="Output .npz path")
    parser.add_argument("--window",     type=int, default=30,   help="Frames per sequence window")
    parser.add_argument("--sample_fps", type=int, default=6,    help="Frames sampled per second")
    args = parser.parse_args()

    X, y, y_score = build_dataset(args.data_dir, args.window, args.sample_fps)
    np.savez(args.out, X=X, y=y, y_score=y_score)
    print(f"\nSaved → {args.out}  ({os.path.getsize(args.out)/1e6:.1f} MB)")
