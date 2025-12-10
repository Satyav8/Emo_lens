# session_system/schemas.py
"""
Schemas for local session logging.
Simple dataclasses used to store session timeline events.
"""

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class SessionEvent:
    timestamp: str
    fused_emotion: Dict[str, Any]
    brain_action: Any

