from dataclasses import dataclass
from datetime import datetime

@dataclass
class SessionEvent:
    timestamp: str
    fused_emotion: dict
    brain_action: dict
