# text_emotion.py
from textblob import TextBlob
from multimodal_emotion.types import Modality

def analyze_text(text):
    if not text or text.strip() == "":
        return None

    polarity = TextBlob(text).sentiment.polarity

    if polarity > 0.3:
        emotion = "positive"
    elif polarity < -0.3:
        emotion = "negative"
    else:
        emotion = "neutral"

    return Modality(
    emotion=emotion,
    confidence=abs(polarity),
    valence=float(polarity),
    arousal=abs(float(polarity)) * 0.5
)

