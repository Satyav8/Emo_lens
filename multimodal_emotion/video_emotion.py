# multimodal_emotion/video_emotion.py
# Improved OpenCV heuristics — more robust scores & confidence smoothing
import cv2
import numpy as np
from PIL import Image
import io
from multimodal_emotion.types import Modality

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
eye_cascade_path = cv2.data.haarcascades + "haarcascade_eye.xml"

face_cascade = cv2.CascadeClassifier(face_cascade_path)
smile_cascade = cv2.CascadeClassifier(smile_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)

VALENCE_MAP = {
    "happy": 0.9, "surprise": 0.4, "neutral": 0.0,
    "sad": -0.6, "angry": -0.9, "fear": -0.7, "disgust": -0.8
}

def _safe_area(rect):
    x,y,w,h = rect
    return max(1, w*h)

def _contrast(gray_face):
    # normalized contrast measure (0..1)
    p2, p98 = np.percentile(gray_face, (2, 98))
    if p98 - p2 <= 0:
        return 0.0
    return float((gray_face.std()) / (p98 - p2 + 1e-6))

def _mouth_open_ratio(face_gray):
    h = face_gray.shape[0]
    start = int(h * 0.5)
    mouth = face_gray[start:h, :]
    if mouth.size == 0:
        return 0.0
    return float(np.mean(mouth) / (np.mean(face_gray) + 1e-6))

def analyze_video_frame(frame_bytes):
    """
    Input: frame bytes from Streamlit camera_input.getvalue()
    Output: Modality(emotion, confidence, valence, arousal)
    """
    try:
        img = Image.open(io.BytesIO(frame_bytes)).convert("RGB")
        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(48,48))

        if len(faces) == 0:
            # No face: low-confidence neutral fallback
            return Modality("neutral", 0.25, 0.0, 0.18)

        # pick largest face
        face = max(faces, key=_safe_area)
        x,y,w,h = face
        face_gray = gray[y:y+h, x:x+w]

        # detect smiles & eyes with tuned params
        smiles = smile_cascade.detectMultiScale(face_gray, scaleFactor=1.6, minNeighbors=18, minSize=(8,8))
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=6, minSize=(8,8))

        # features
        smile_count = len(smiles)
        eye_count = len(eyes)
        mouth_ratio = _mouth_open_ratio(face_gray)
        contrast = _contrast(face_gray)

        # signals (0..1)
        smile_signal = min(1.0, smile_count * 0.7 + mouth_ratio * 0.3)
        surprise_signal = min(1.0, eye_count * 0.5 + mouth_ratio * 0.6 + contrast * 0.2)
        neutral_signal = 0.15
        sad_signal = max(0.0, 0.25 - smile_signal) * 0.8
        angry_signal = max(0.0, 0.2 - smile_signal) * 0.6
        fear_signal = max(0.0, 0.2 - smile_signal) * (1 - eye_count*0.2)
        disgust_signal = 0.02

        scores = {
            "happy": smile_signal,
            "surprise": surprise_signal,
            "neutral": neutral_signal,
            "sad": sad_signal,
            "angry": angry_signal,
            "fear": fear_signal,
            "disgust": disgust_signal
        }

        # soft-normalize & smoothing
        total = sum(scores.values()) + 1e-9
        for k in scores:
            scores[k] = float(scores[k] / total)

        emotion = max(scores, key=lambda k: scores[k])
        raw_conf = float(scores[emotion])

        # confidence boost if multiple cues agree
        cue_strength = (smile_signal + surprise_signal + (eye_count/2.0)) / 3.0
        confidence = min(1.0, raw_conf * 0.9 + cue_strength * 0.2)

        valence = VALENCE_MAP.get(emotion, 0.0)
        arousal = min(1.0, 0.2 + confidence * 0.9)

        return Modality(emotion=emotion, confidence=confidence, valence=valence, arousal=arousal)

    except Exception as e:
        print("video_emotion ERROR:", e)
        return Modality("neutral", 0.2, 0.0, 0.2)









