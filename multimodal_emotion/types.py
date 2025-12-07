# multimodal_emotion/types.py
from dataclasses import dataclass, asdict

# --------------------------------------------
# Base unit returned by video/audio/text models
# --------------------------------------------
@dataclass
class Modality:
    emotion: str = None
    confidence: float = 0.0
    valence: float = 0.0
    arousal: float = 0.0

    # dict-like access
    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return asdict(self)

    def items(self):
        return self.to_dict().items()

    def __repr__(self):
        return f"Modality(emotion={self.emotion}, conf={self.confidence:.2f}, valence={self.valence:.2f}, arousal={self.arousal:.2f})"


# --------------------------------------------
# Fusion helper: represent one modality’s score
# --------------------------------------------
@dataclass
class ModalityScore:
    emotion: str
    confidence: float
    valence: float
    arousal: float

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------
# Final fused emotional state returned by fusion.py
# --------------------------------------------
@dataclass
class EmotionVector:
    final_emotion: str
    valence: float
    arousal: float
    confidence: float
    modalities: dict  # { "video": ModalityScore, "text": ModalityScore, ... }

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def to_dict(self):
        # convert sub-modalities too
        return {
            "final_emotion": self.final_emotion,
            "valence": self.valence,
            "arousal": self.arousal,
            "confidence": self.confidence,
            "modalities": {
                k: v.to_dict() if hasattr(v, "to_dict") else v
                for k, v in self.modalities.items()
            },
        }


