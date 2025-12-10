# multimodal_emotion/fusion.py
"""
Robust fusion function that accepts:
- Modality objects (with attributes and .to_dict())
- Plain dicts
- None for missing modalities

Returns:
- EmotionVector (from multimodal_emotion.types) or None
"""

from multimodal_emotion.types import EmotionVector, ModalityScore

def fuse(video=None, audio=None, text=None):
    modalities = {}
    vals, aros, confs = [], [], []
    label_votes = {}

    def extract(m):
        """
        Normalize modality to a simple dict with keys:
        emotion, confidence, valence, arousal
        """
        if m is None:
            return None

        # Modality-like object with attributes
        if hasattr(m, "emotion") and hasattr(m, "confidence"):
            try:
                return {
                    "emotion": getattr(m, "emotion"),
                    "confidence": float(getattr(m, "confidence") or 0.0),
                    "valence": float(getattr(m, "valence") or 0.0),
                    "arousal": float(getattr(m, "arousal") or 0.0),
                }
            except Exception:
                # fallback to dictionary-style extraction
                pass

        # If dict-like
        if isinstance(m, dict):
            return {
                "emotion": m.get("emotion"),
                "confidence": float(m.get("confidence") or 0.0),
                "valence": float(m.get("valence") or 0.0),
                "arousal": float(m.get("arousal") or 0.0),
            }

        # Unknown type
        return None

    def add(m, name):
        norm = extract(m)
        if not norm or not norm.get("emotion"):
            return

        ms = ModalityScore(
            emotion=norm["emotion"],
            confidence=norm["confidence"],
            valence=norm["valence"],
            arousal=norm["arousal"]
        )

        modalities[name] = ms
        vals.append(ms.valence * ms.confidence)
        aros.append(ms.arousal * ms.confidence)
        confs.append(ms.confidence)
        label_votes[ms.emotion] = label_votes.get(ms.emotion, 0.0) + ms.confidence

    add(video, "video")
    add(audio, "audio")
    add(text, "text")

    # If no valid modalities, return None
    if not modalities:
        return None

    total_conf = sum(confs) if confs else 1.0
    final_emotion = max(label_votes.items(), key=lambda x: x[1])[0]

    # Weighted averages
    valence = (sum(vals) / total_conf) if total_conf else 0.0
    arousal = (sum(aros) / total_conf) if total_conf else 0.0
    # Confidence averaged across modalities (0..1)
    confidence = (total_conf / len(confs)) if confs else 0.0

    return EmotionVector(
        final_emotion=final_emotion,
        valence=valence,
        arousal=arousal,
        confidence=confidence,
        modalities=modalities
    )
