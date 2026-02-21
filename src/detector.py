import os
import csv
import cv2
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict

from .features import to_gray_small, mean_abs_diff, sharpness_laplacian, optical_flow_stats
from .viz import overlay_text, border
from .report import build_dashboard
from .ai_model import train_classifier, predict, int_to_label

def robust_threshold(x: np.ndarray, k: float) -> float:
    # median + k*MAD
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-9
    return med + k * mad

def extract_features(frames_bgr: List[np.ndarray], fps: float, resize: Tuple[int,int]) -> pd.DataFrame:
    n = len(frames_bgr)
    sigs = [to_gray_small(fr, resize) for fr in frames_bgr]

    diff_prev = np.zeros(n, dtype=np.float32)
    sharp = np.zeros(n, dtype=np.float32)
    flow_mean = np.zeros(n, dtype=np.float32)
    flow_p95 = np.zeros(n, dtype=np.float32)

    for i in range(n):
        sharp[i] = sharpness_laplacian(sigs[i])
        if i >= 1:
            diff_prev[i] = mean_abs_diff(sigs[i-1], sigs[i])
            fs = optical_flow_stats(sigs[i-1], sigs[i])
            flow_mean[i] = fs["flow_mean"]
            flow_p95[i] = fs["flow_p95"]

    # diff_next is just shifted diff_prev
    diff_next = np.zeros(n, dtype=np.float32)
    diff_next[:-1] = diff_prev[1:]

    df = pd.DataFrame({
        "frame_idx": np.arange(n, dtype=np.int32),
        "time_sec": np.arange(n, dtype=np.float32) / float(max(fps, 1e-6)),
        "fps": float(fps),
        "diff_prev": diff_prev,
        "diff_next": diff_next,
        "sharpness": sharp,
        "flow_mean": flow_mean,
        "flow_p95": flow_p95,
    })
    return df

def heuristic_labels(df: pd.DataFrame) -> np.ndarray:
    # Drop: spike in diff_prev OR flow_mean
    dp = df["diff_prev"].to_numpy()
    fm = df["flow_mean"].to_numpy()
    sh = df["sharpness"].to_numpy()

    # robust thresholds
    dp_thr = robust_threshold(dp[1:], k=6.0) if len(dp) > 2 else float("inf")
    fm_thr = robust_threshold(fm[1:], k=6.0) if len(fm) > 2 else float("inf")
    merge_low = robust_threshold(dp[1:], k=2.5) if len(dp) > 2 else float("inf")

    sharp_med = float(np.median(sh))
    sharp_thr = sharp_med * 0.75

    labels = np.array(["Normal"] * len(df), dtype=object)

    for i in range(len(df)):
        if i >= 1 and (dp[i] >= dp_thr or fm[i] >= fm_thr):
            labels[i] = "Frame Drop"

    # Merge: medium-high diffs both sides, similar magnitude, and low sharpness
    for i in range(1, len(df)-1):
        a = float(df.loc[i, "diff_prev"])
        b = float(df.loc[i, "diff_next"])
        if a >= merge_low and b >= merge_low and labels[i] != "Frame Drop":
            rel = abs(a-b) / max(a,b,1e-9)
            if rel <= 0.20 and float(df.loc[i, "sharpness"]) <= sharp_thr:
                labels[i] = "Frame Merge"

    return labels

def make_synthetic_dataset(frames_bgr: List[np.ndarray], fps: float, resize: Tuple[int,int], seed: int = 42,
                           n_samples: int = 600) -> Tuple[np.ndarray, np.ndarray]:
    # Create synthetic (feature, label) data using the input video itself.
    # - Normal: consecutive frames
    # - Drop: skip one or more frames (simulates missing)
    # - Merge: blend two consecutive frames (ghosting)
    rng = np.random.default_rng(seed)
    n = len(frames_bgr)
    if n < 10:
        raise RuntimeError("Video too short for AI mode (need >= 10 frames).")

    X_list = []
    y_list = []

    def feat_from_pair(prev_fr, fr, next_fr) -> np.ndarray:
        # compute local 1-frame features similar to runtime
        prev_g = to_gray_small(prev_fr, resize)
        g = to_gray_small(fr, resize)
        next_g = to_gray_small(next_fr, resize)
        dp = mean_abs_diff(prev_g, g)
        dn = mean_abs_diff(g, next_g)
        sh = sharpness_laplacian(g)
        fs = optical_flow_stats(prev_g, g)
        return np.array([dp, dn, sh, fs["flow_mean"], fs["flow_p95"]], dtype=np.float32)

    # Normal samples
    for _ in range(n_samples//3):
        i = int(rng.integers(1, n-2))
        X_list.append(feat_from_pair(frames_bgr[i-1], frames_bgr[i], frames_bgr[i+1]))
        y_list.append("Normal")

    # Drop samples: current frame is far from previous (skipping)
    for _ in range(n_samples//3):
        i = int(rng.integers(3, n-2))
        skip = int(rng.integers(2, min(8, i)))  # skip 2..
        X_list.append(feat_from_pair(frames_bgr[i-skip], frames_bgr[i], frames_bgr[i+1]))
        y_list.append("Frame Drop")

    # Merge samples: blend current as alpha*prev + (1-alpha)*curr, which creates ghosting-ish frame
    for _ in range(n_samples - 2*(n_samples//3)):
        i = int(rng.integers(2, n-2))
        alpha = float(rng.uniform(0.35, 0.65))
        merged = cv2.addWeighted(frames_bgr[i-1], alpha, frames_bgr[i], 1.0-alpha, 0)
        X_list.append(feat_from_pair(frames_bgr[i-1], merged, frames_bgr[i+1]))
        y_list.append("Frame Merge")

    X = np.vstack(X_list)
    y = np.array(y_list, dtype=object)
    return X, y

def ai_labels(df: pd.DataFrame, frames_bgr: List[np.ndarray], fps: float, resize: Tuple[int,int], seed: int) -> np.ndarray:
    # Train on synthetic features (no external dataset) then predict on real df
    X_train, y_train = make_synthetic_dataset(frames_bgr, fps, resize, seed=seed, n_samples=600)
    model = train_classifier(X_train, y_train, seed=seed)

    X = df[["diff_prev","diff_next","sharpness","flow_mean","flow_p95"]].to_numpy(dtype=np.float32)
    y_pred = predict(model, X)
    return y_pred

def write_outputs(frames_bgr: List[np.ndarray], df: pd.DataFrame, labels: np.ndarray, out_dir: str, report_dir: str,
                  input_fps: float):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    out_csv = os.path.join(out_dir, "report.csv")
    df_out = df.copy()
    df_out["label"] = labels
    df_out.to_csv(out_csv, index=False)

    # Annotated video
    h, w = frames_bgr[0].shape[:2]
    out_video = os.path.join(out_dir, "annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(out_video, fourcc, float(input_fps), (w, h))

    for i, frame in enumerate(frames_bgr):
        lab = str(labels[i])
        txt = f"Frame {i} | {lab} | diff={df_out.loc[i,'diff_prev']:.2f} flow={df_out.loc[i,'flow_mean']:.2f} sharp={df_out.loc[i,'sharpness']:.0f}"
        out = overlay_text(frame, txt)
        out = border(out, lab)
        vw.write(out)

    vw.release()

    # HTML dashboard + charts
    build_dashboard(df_out, report_dir, annotated_video_rel=os.path.join(out_dir, "annotated.mp4"))

    return out_video, out_csv

def run_detection(input_path: str, mode: str, out_dir: str, report_dir: str, resize: Tuple[int,int],
                  max_frames: Optional[int] = None, seed: int = 42):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0

    frames = []
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        frames.append(fr)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if len(frames) < 3:
        raise RuntimeError("Video too short (need >= 3 frames).")

    df = extract_features(frames, fps, resize)

    if mode == "heuristic":
        labels = heuristic_labels(df)
    else:
        labels = ai_labels(df, frames, fps, resize, seed=seed)

    out_video, out_csv = write_outputs(frames, df, labels, out_dir, report_dir, input_fps=fps)

    # console summary
    normal = int((labels=="Normal").sum())
    drops = int((labels=="Frame Drop").sum())
    merges = int((labels=="Frame Merge").sum())
    print("Done.")
    print(f"Frames: {len(frames)} | Normal: {normal} | Drops: {drops} | Merges: {merges}")
    print(f"CSV: {out_csv}")
    print(f"Annotated video: {out_video}")
    print(f"Dashboard: {os.path.join(report_dir,'dashboard.html')}")
