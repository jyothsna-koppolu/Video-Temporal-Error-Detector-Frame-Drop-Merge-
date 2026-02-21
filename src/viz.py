import cv2
import numpy as np

def overlay_text(frame_bgr: np.ndarray, text: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    bar_h = 64
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    out = cv2.addWeighted(overlay, 0.45, out, 0.55, 0)
    cv2.putText(out, text, (15, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2, cv2.LINE_AA)
    return out

def border(frame_bgr: np.ndarray, label: str) -> np.ndarray:
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    t = max(6, int(min(h,w)*0.01))
    if label == "Frame Drop":
        color = (0,0,255)
    elif label == "Frame Merge":
        color = (0,165,255)
    else:
        color = (0,255,0)
    cv2.rectangle(out, (0,0), (w-1,h-1), color, t)
    return out
