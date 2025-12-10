# multimodal_emotion/audio_emotion.py
# Lightweight, robust audio emotion detector using signal features (librosa)
import io
import numpy as np
import librosa
from multimodal_emotion.types import Modality

# emotion labels we support
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

VALENCE_MAP = {
    "happy": 0.9, "surprised": 0.4, "calm": 0.25, "neutral": 0.0,
    "sad": -0.6, "fearful": -0.7, "angry": -0.85, "disgust": -0.8
}

def _rms_energy(y):
    return float(np.mean(librosa.feature.rms(y=y)))

def _zcr(y):
    return float(np.mean(librosa.feature.zero_crossing_rate(y)))

def _spectral_centroid(y, sr):
    return float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))

def _pitch_confidence(y, sr):
    # use short pitch estimation; return (pitch, confidence)
    try:
        f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
        # remove nans
        f0 = f0[~np.isnan(f0)]
        if len(f0) == 0:
            return 0.0, 0.0
        return float(np.median(f0)), float(np.mean(f0) > 0)
    except Exception:
        return 0.0, 0.0

def analyze_audio(audio_bytes):
    """
    Input: raw audio bytes (wav/mp3). Output: Modality
    """
    try:
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000, mono=True)

        # core features
        rms = _rms_energy(y)               # energy
        zcr = _zcr(y)                      # voice activity / noisiness
        centroid = _spectral_centroid(y, sr)
        pitch, pitch_conf = _pitch_confidence(y, sr)

        # MFCC summary
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        mfcc_score = float(np.mean(np.abs(mfcc_mean)) / (np.mean(np.abs(mfcc_std)) + 1e-6))

        # Heuristic rules to map into emotion signals
        # energy + high centroid -> excited / angry / surprised
        excited_signal = min(1.0, rms * 6.0 + (centroid / (sr/2)) * 2.0)
        calm_signal = max(0.0, 1.0 - excited_signal) * 0.6
        happy_signal = min(1.0, pitch_conf * 0.9 + mfcc_score * 0.3)
        sad_signal = max(0.0, 0.6 - rms) * 1.2
        angry_signal = min(1.0, max(0.0, rms * 4.0 - 0.2))
        fear_signal = max(0.0, 0.4 - pitch_conf)
        disgust_signal = 0.05
        neutral_signal = 0.15

        signals = {
            "happy": happy_signal,
            "surprised": excited_signal * 0.65,
            "calm": calm_signal,
            "neutral": neutral_signal,
            "sad": sad_signal,
            "angry": angry_signal,
            "fearful": fear_signal,
            "disgust": disgust_signal
        }

        # normalize
        total = sum(signals.values()) + 1e-9
        for k in signals:
            signals[k] = float(signals[k] / total)

        emotion = max(signals, key=lambda k: signals[k])
        raw_conf = float(signals[emotion])

        # smooth confidence with energy/pitch cues
        confidence = min(0.9999, raw_conf * 0.9 + (rms * 0.4) + (pitch_conf * 0.3))

        valence = VALENCE_MAP.get(emotion, 0.0)
        arousal = min(1.0, 0.2 + confidence * 0.9)

        return Modality(emotion=emotion, confidence=confidence, valence=valence, arousal=arousal)

    except Exception as e:
        print("audio_emotion ERROR:", e)
        return Modality("neutral", 0.25, 0.0, 0.15)

