# assistant_engine/generator.py
import os
from assistant_engine.policy import pick_style, TEACHING_STYLES

def build_prompt(user_query, emotion_state, brain_state):
    style_key = pick_style(
        getattr(emotion_state, "final_emotion", None),
        brain_state.get("cognitive_load"),
        brain_state.get("predicted_state")
    )
    style_instructions = TEACHING_STYLES.get(style_key, "")

    return f"""
You are EmoLens — an adaptive, empathetic AI teacher.

STUDENT EMOTIONAL STATE
- Emotion: {emotion_state.final_emotion}
- Valence: {emotion_state.valence:.2f}
- Arousal: {emotion_state.arousal:.2f}
- Engagement: {brain_state['engagement_level']}
- Cognitive Load: {brain_state['cognitive_load']}
- Predicted State: {brain_state['predicted_state']}

TEACHING STYLE SELECTED → {style_key.upper()}
STYLE RULES → {style_instructions}

Task:
Provide:
1) A clear, simple explanation tailored to the student's state.
2) A short encouragement aligned with their emotion.
3) A next step or question to continue learning.

STUDENT QUESTION:
{user_query}
"""

def call_llm(prompt):
    """
    Uses ONLY the new OpenAI Python SDK style.
    If OPENAI_API_KEY is missing, returns a mock response.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if not api_key:
        return (
            "[Local Mock Response]\n\n"
            "1) Explanation: Here's the concept simplified.\n"
            "2) Encouragement: You're doing great — keep going!\n"
            "3) Next step: Try a tiny example to reinforce this.\n"
        )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=250,
            temperature=0.5
        )

        # NEW SDK: message content is an attribute, NOT a dict
        return response.choices[0].message.content

    except Exception as e:
        return f"[LLM ERROR] {e}"


def generate_teaching_reply(user_query, emotion_state, brain_state):
    prompt = build_prompt(user_query, emotion_state, brain_state)
    return call_llm(prompt)

