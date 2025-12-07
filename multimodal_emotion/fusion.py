# fusion.py
from types import SimpleNamespace
from multimodal_emotion.types import EmotionVector, ModalityScore

def fuse(video=None, audio=None, text=None):
    modalities = {}
    vals, aros, confs = [], [], []
    label_votes = {}

    def add(m, name):
        if not m: return
        ms = ModalityScore(
            emotion=m["emotion"],
            confidence=m["confidence"],
            valence=m["valence"],
            arousal=m["arousal"]
        )
        modalities[name] = ms
        vals.append(ms.valence * ms.confidence)
        aros.append(ms.arousal * ms.confidence)
        confs.append(ms.confidence)
        label_votes[ms.emotion] = label_votes.get(ms.emotion, 0) + ms.confidence

    add(video, "video")
    add(audio, "audio")
    add(text, "text")

    if not modalities:
        return None

    final_emotion = max(label_votes.items(), key=lambda x: x[1])[0]
    total_conf = sum(confs)

    return EmotionVector(
        final_emotion=final_emotion,
        valence=sum(vals)/total_conf,
        arousal=sum(aros)/total_conf,
        confidence=total_conf / len(confs),
        modalities=modalities
    )
