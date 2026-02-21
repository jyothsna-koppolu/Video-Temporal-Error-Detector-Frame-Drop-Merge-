import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

LABELS = ["Normal", "Frame Drop", "Frame Merge"]

@dataclass
class AIModel:
    pipe: Pipeline

def train_classifier(X: np.ndarray, y: np.ndarray, seed: int = 42) -> AIModel:
    # Simple, fast, reliable baseline for hackathons
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=seed))
    ])
    pipe.fit(X, y)
    return AIModel(pipe=pipe)

def predict(model: AIModel, X: np.ndarray) -> np.ndarray:
    return model.pipe.predict(X)

def label_to_int(labels: np.ndarray) -> np.ndarray:
    mapping = {name:i for i,name in enumerate(LABELS)}
    return np.array([mapping[l] for l in labels], dtype=np.int32)

def int_to_label(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.int32)
    return np.array([LABELS[i] for i in y], dtype=object)
