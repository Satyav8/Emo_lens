# multimodal_emotion/video_emotion.py
# Lightweight, robust face-emotion heuristic using OpenCV Haar cascades.
# video_emotion.py
from fer import FER
import numpy as np
from PIL import Image
import io
from multimodal_emotion.types import Modality

_detector = FER(mtcnn=True)

VALENCE_MAP = {
    "happy": 1, "surprise": 0.5,
    "neutral": 0,
    "sad": -0.5, "fear": -0.7, "angry": -1, "disgust": -0.8
}

def analyze_video_frame(frame_bytes):
    img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
    arr = np.array(img)

    res = _detector.detect_emotions(arr)
    if not res:
        return None

    best = max(res, key=lambda r: r["box"][2] * r["box"][3])
    emotion, conf = max(best["emotions"].items(), key=lambda x: x[1])

    # simple valence & arousal estimation
    valence = VALENCE_MAP.get(emotion, 0)
    arousal = abs(conf)  

    return Modality(
    emotion=emotion,
    confidence=float(conf),
    valence=valence,
    arousal=arousal
)




