import uuid
import json
from datetime import datetime
from session_system.schemas import SessionEvent
from session_system.db import get_db


class SessionManager:
    def __init__(self):
        self.session_id = None
        self.events = []
        self.db = get_db()

    # -------------------------
    # Log to Supabase database
    # -------------------------
    def log_to_db(self, fused_emotion, brain_output):
        if not self.db:
            return  # database not configured

        payload = {
            "session_id": self.session_id,
            "emotion": fused_emotion.final_emotion,
            "valence": fused_emotion.valence,
            "arousal": fused_emotion.arousal,
            "confidence": fused_emotion.confidence,
            "brain_action": brain_output["recommended_action"],
            "micro_prompt": brain_output["micro_prompt"],
            "timestamp": datetime.now().isoformat(),
            "modalities": {
                k: vars(v) for k, v in fused_emotion.modalities.items()
            }
        }

        try:
            self.db.table("emotion_logs").insert(payload).execute()
        except Exception as e:
            print("DB LOGGING ERROR:", e)

    # -------------------------
    # Start a new session
    # -------------------------
    def start_session(self):
        self.session_id = str(uuid.uuid4())
        self.events = []
        return self.session_id

    # -------------------------
    # Log event locally
    # -------------------------
    def log(self, fused_emotion, brain_output):
        if not self.session_id:
            self.start_session()

        entry = SessionEvent(
            timestamp=datetime.now().isoformat(),
            fused_emotion={
                "emotion": fused_emotion.final_emotion,
                "valence": fused_emotion.valence,
                "arousal": fused_emotion.arousal,
                "confidence": fused_emotion.confidence,
                "modalities": {k: (v.to_dict() if hasattr(v, "to_dict") else dict(v)) for k, v in fused_emotion.modalities.items()}

            },
            brain_action=brain_output
        )

        self.events.append(entry)

        # Log to DB as well
        self.log_to_db(fused_emotion, brain_output)

    # -------------------------
    # Return local session timeline
    # -------------------------
    def get_timeline(self):
        return self.events

    # -------------------------
    # Save session locally as JSON
    # -------------------------
    def save_local(self):
        if not self.session_id:
            return None

        path = f"session_{self.session_id}.json"
        data = [vars(e) for e in self.events]

        with open(path, "w") as f:
            json.dump(data, f, indent=4)

        return path

    # -------------------------
    # End session
    # -------------------------
    def end_session(self):
        path = self.save_local()
        self.session_id = None
        self.events = []
        return path


# Singleton instance
session_manager = SessionManager()

