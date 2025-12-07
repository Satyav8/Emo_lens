from multimodal_brain.rules import THRESHOLDS
from multimodal_brain.utils import EmotionHistory

history = EmotionHistory(maxlen=10)

def analyze_state(ev):
    if not ev:
        return {
            "engagement_level": "none",
            "cognitive_load": "unknown",
            "predicted_state": "unknown",
            "recommended_action": "await_input",
            "micro_prompt": "Start whenever you're ready!"
        }

    history.add(ev.valence, ev.arousal)

    v = ev.valence
    a = ev.arousal
    emo = ev.final_emotion

    # 1. ENGAGEMENT
    if v > THRESHOLDS["high_engagement_valence"]:
        engagement = "high"
    elif v < THRESHOLDS["low_engagement_valence"]:
        engagement = "low"
    else:
        engagement = "medium"

    # 2. COGNITIVE LOAD
    if a < 0.25:
        load = "low"
    elif a < 0.55:
        load = "medium"
    else:
        load = "high"

    # 3. Momentum / future prediction
    momentum_val = history.momentum()
    arousal_trend = history.arousal_trend()

    if momentum_val < -0.03 and arousal_trend > 0.03:
        predicted = "incoming_frustration"
    elif momentum_val > 0.04:
        predicted = "improving"
    else:
        predicted = "stable"

    # 4. Teaching action
    if predicted == "incoming_frustration":
        action = "slow_down"
        micro = "I notice this might be getting tricky. Want a simpler version?"
    elif emo in ("anger", "fear", "sadness"):
        action = "supportive_recap"
        micro = "Let’s take this step-by-step. I’ve got you."
    elif engagement == "high":
        action = "advance"
        micro = "You're doing great—want to try something harder?"
    elif engagement == "low":
        action = "re_engage"
        micro = "Should I show an example or switch style?"
    else:
        action = "continue"
        micro = "Let's go ahead at this pace."

    return {
        "engagement_level": engagement,
        "cognitive_load": load,
        "predicted_state": predicted,
        "recommended_action": action,
        "micro_prompt": micro
    }
