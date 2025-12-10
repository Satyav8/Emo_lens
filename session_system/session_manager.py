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
    # Convert modality objects safely
    # -------------------------
    def _serialize_modalities(self, modalities):
        out = {}
        for k, v in modalities.items():
            if hasattr(v, "to_dict"):
                out[k] = v.to_dict()
            elif hasattr(v, "__dict__"):
                out[k] = v.__dict__
            else:
                out[k] = v
        return out

    # -------------------------
    # Log to Supabase database
    # -------------------------
    def log_to_db(self, fused_emotion, brain_output):
        if not self.db:
            return  # database not configured

        payload = {
            "session_id": self.session_id,
            "emotion": fused_emotion.final_emotion,
            "valence": float(fused_emotion.valence),
            "arousal": float(fused_emotion.arousal),
            "confidence": float(fused_emotion.confidence),
            "brain_action": brain_output.get("recommended_action"),
            "micro_prompt": brain_output.get("micro_prompt"),
            "timestamp": datetime.now().isoformat(),

            # Must be JSON serializable for Supabase
            "modalities": self._serialize_modalities(fused_emotion.modalities)
        }

        try:
            self.db.table("emotion_logs").insert(payload).execute()
        except Exception as e:
            print("\n🚨 DB LOGGING ERROR:", e)
            print("Payload that failed:", payload)

    # -------------------------
    # Start a new session
    # -------------------------
    def start_session(self):
        self.session_id = str(uuid.uuid4())
        self.events = []
        return self.session_id

    # -------------------------
    # Log event locally + DB
    # -------------------------
    def log(self, fused_emotion, brain_output):
        if not self.session_id:
            self.start_session()

        entry = SessionEvent(
            timestamp=datetime.now().isoformat(),
            fused_emotion={
                "emotion": fused_emotion.final_emotion,
                "valence": float(fused_emotion.valence),
                "arousal": float(fused_emotion.arousal),
                "confidence": float(fused_emotion.confidence),

                # serialized modalities
                "modalities": self._serialize_modalities(fused_emotion.modalities)
            },
            brain_action=brain_output
        )

        self.events.append(entry)

        # Log to DB (RLS safe)
        self.log_to_db(fused_emotion, brain_output)

    # -------------------------
    # Return local session timeline
    # -------------------------
    def get_timeline(self):
        return self.events

    # -------------------------
    # Save session to a local JSON file
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


