# assistant_engine/policy.py
TEACHING_STYLES = {
    "supportive": "Warm, encouraging, slow-paced, simple language. Short steps and examples.",
    "explainer":   "Structured step-by-step explanation with a concise example.",
    "motivational":"Energetic, praise-oriented with a small challenge to keep momentum.",
    "expert":      "Concise, technical, assume base knowledge and highlight principles."
}

def pick_style(emotion_label, cognitive_load, predicted_state):
    emotion = (emotion_label or "").lower()
    # highest-priority conditions
    if predicted_state == "incoming_frustration" or cognitive_load == "high":
        return "supportive"
    if emotion in ("anger", "fear", "sadness"):
        return "supportive"
    if emotion in ("happiness", "positive"):
        return "motivational"
    if emotion in ("neutral", "surprise") or cognitive_load == "medium":
        return "explainer"
    return "explainer"

