import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

@dataclass
class FrameFeatures:
    idx: int
    time_sec: float
    diff_prev: float
    diff_next: float
    sharpness: float
    flow_mean: float
    flow_p95: float

def to_gray_small(frame_bgr: np.ndarray, size: Tuple[int,int]) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)

def mean_abs_diff(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(cv2.absdiff(a, b)))

def sharpness_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def optical_flow_stats(prev_gray: np.ndarray, gray: np.ndarray) -> Dict[str,float]:
    # Farnebäck dense optical flow
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1], angleInDegrees=False)
    mag = mag.astype(np.float32)
    mean = float(np.mean(mag))
    p95 = float(np.percentile(mag, 95))
    return {"flow_mean": mean, "flow_p95": p95}
