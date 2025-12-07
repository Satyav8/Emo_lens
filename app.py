#############################
#   EmoLens — Final Draft   #
#     Student + Dashboard   #
#############################


from dotenv import load_dotenv
load_dotenv()  # MUST be first

import os
os.system("pip install fer==20.0.0 mtcnn tensorflow==2.12.0")

import os
import json
import streamlit as st
from datetime import datetime
import pandas as pd

# Multimodal Emotion Engine
from multimodal_emotion.video_emotion import analyze_video_frame
from multimodal_emotion.audio_emotion import analyze_audio
from multimodal_emotion.text_emotion import analyze_text
from multimodal_emotion.fusion import fuse

# Adaptive Learning Brain
from multimodal_brain.brain import analyze_state

# Session Manager + DB
from session_system.session_manager import session_manager
from session_system.db import get_db

# LLM generator
from assistant_engine.generator import generate_teaching_reply

# ---- Config ----
st.set_page_config(page_title="EmoLens — Live + Dashboard", layout="wide")
DB = get_db()

# --------------------------
# Theme / small animations
# --------------------------
st.markdown(
    """
<style>
/* pulsing metric */
@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.04); }
  100% { transform: scale(1); }
}
div[data-testid="stMetricValue"] { animation: pulse 2.4s infinite; }

/* subtle card style */
.emolens-card {
  padding:14px;
  border-radius:10px;
  background:#ffffff;
  border:1px solid #eef2ff;
  box-shadow: 0 6px 18px rgba(15,23,42,0.03);
}
.small-muted { color:#6b7280; font-size:13px; }
</style>
""",
    unsafe_allow_html=True,
)

# Branding header
st.markdown(
    """
<div style="text-align:center;margin-bottom:14px;">
  <h1 style="color:#4f46e5;margin:0;">EmoLens</h1>
  <div style="color:#6b7280;">Emotion-aware adaptive learning assistant</div>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Helper: Emotion Orb ----------
def emotion_orb(valence, arousal, size=128):
    # clamp inputs
    try:
        v = max(-1.0, min(1.0, float(valence)))
        a = max(0.0, min(1.0, float(arousal)))
    except Exception:
        v, a = 0.0, 0.0
    # Map valence (-1..1) -> hue (red (0) to green (120))
    hue = int((v + 1) / 2 * 120)
    # brightness from arousal
    lightness = int(40 + a * 45)
    color = f"hsl({hue}, 78%, {lightness}%)"
    orb_html = f"""
    <div style="display:flex;align-items:center;flex-direction:column">
      <div style="
        width:{size}px;height:{size}px;
        border-radius:50%;
        background:{color};
        box-shadow:0 8px 30px {color}55;
        margin-bottom:8px;
        transition: all 300ms ease;">
      </div>
    </div>
    """
    return orb_html


#########################
#   Sidebar Navigation   #
#########################
st.sidebar.title("EmoLens")
page = st.sidebar.radio("Mode", ["Student (Live)", "Educator Dashboard", "DB Test / Admin"])

#################################
#   Database Helper Functions   #
#################################
def fetch_latest_rows(limit=50):
    if not DB:
        return []
    try:
        resp = DB.table("emotion_logs").select("*").order("timestamp", desc=True).limit(limit).execute()
        return resp.data or []
    except Exception as e:
        st.error(f"DB Fetch Error: {e}")
        return []

def fetch_sessions(limit=100):
    rows = fetch_latest_rows(limit=200)
    sessions = {}
    for r in rows:
        sid = r.get("session_id")
        ts = r.get("timestamp")
        if sid:
            if sid not in sessions or ts > sessions[sid]["timestamp"]:
                sessions[sid] = {"session_id": sid, "timestamp": ts}
    out = sorted(sessions.values(), key=lambda x: x["timestamp"], reverse=True)
    return out[:limit]

def fetch_session_rows(session_id, limit=1000):
    if DB:
        try:
            resp = DB.table("emotion_logs").select("*").eq("session_id", session_id).order("timestamp", desc=False).limit(limit).execute()
            return resp.data or []
        except Exception:
            pass
    path = f"session_{session_id}.json"
    if os.path.exists(path):
        return json.load(open(path, "r"))
    return []

###############################
#   Analytics Helper Methods   #
###############################
def compute_session_metrics(rows):
    if not rows:
        return {}
    valences, arousals, emotions = [], [], {}
    for r in rows:
        fe = r.get("fused_emotion") or {}
        v = fe.get("valence", r.get("valence", 0))
        a = fe.get("arousal", r.get("arousal", 0))
        e = fe.get("emotion", r.get("emotion", "unknown"))
        valences.append(v)
        arousals.append(a)
        emotions[e] = emotions.get(e, 0) + 1
    return {
        "avg_valence": sum(valences)/len(valences) if valences else 0,
        "avg_arousal": sum(arousals)/len(arousals) if arousals else 0,
        "dominant_emotion": max(emotions, key=emotions.get) if emotions else "n/a",
        "events": len(rows)
    }

def detect_spikes(rows, valence_drop=-0.4, arousal_rise=0.5):
    spikes = []
    for i, r in enumerate(rows):
        fe = r.get("fused_emotion") or {}
        v = fe.get("valence", 0)
        a = fe.get("arousal", 0)
        if v <= valence_drop and a >= arousal_rise:
            spikes.append((i, r.get("timestamp")))
    return spikes

###############################
#   PAGE 1 — Student (Live)   #
###############################
if page == "Student (Live)":
    st.header("Student — Live Interaction")

    # set containers and default variables
    col_left, col_right = st.columns([2, 1])
    fusion = None
    brain_out = None
    ai_reply = None

    # Input widgets
    with col_left:
        camera_bytes = st.camera_input("Take a photo")
        audio_file = st.file_uploader("Upload audio (optional)", type=["wav", "mp3", "m4a"])
        text_input = st.text_area("Or type how you feel / what are you learning?", height=90)
        user_query = st.text_input("Ask a question / request an explanation (optional):", value="")

    with col_right:
        st.markdown("### Live Status")
        status_placeholder = st.empty()
        st.markdown("---")
        st.markdown("### Suggested Micro-Action")
        action_placeholder = st.empty()

    # Process inputs (safe calls)
    video_res = None
    audio_res = None
    text_res = None

    if camera_bytes:
        try:
            video_res = analyze_video_frame(camera_bytes.getvalue())
        except Exception as e:
            st.warning("Video analyze error: " + str(e))

    if audio_file:
        try:
            audio_res = analyze_audio(audio_file.getvalue())
        except Exception as e:
            st.warning("Audio analyze error: " + str(e))

    if text_input and text_input.strip():
        try:
            text_res = analyze_text(text_input)
        except Exception as e:
            st.warning("Text analyze error: " + str(e))

    # Fuse modalities
    try:
        fusion = fuse(video=video_res, audio=audio_res, text=text_res)
    except Exception as e:
        fusion = None
        st.warning("Fusion error: " + str(e))


    if fusion:
        brain_out = analyze_state(fusion)
        session_manager.log(fusion, brain_out)

    # LLM Reply (only when user asked)
    if fusion and user_query and user_query.strip():
        try:
            ai_reply = generate_teaching_reply(user_query.strip(), fusion, brain_out)
        except Exception as e:
            ai_reply = f"[LLM ERROR] {e}"

    # Suggest action function
    def suggest_action(ev):
        if not ev:
            return "Provide camera/audio/text to begin."
        v, a, emo = ev.valence, ev.arousal, ev.final_emotion
        if v <= -0.4 and a > 0.4:
            return "Detected frustration → slow down & simplify."
        if v <= -0.3:
            return "Low valence → offer supportive recap."
        if v >= 0.6:
            return "Positive → increase challenge."
        if emo in ("surprise", "neutral"):
            return "Ask a clarifying question."
        return "Continue at the same pace."

    # Display outputs — polished UI
    st.markdown("## 🎭 Emotional Snapshot")
    if fusion:
        status_placeholder.markdown(
        f"**{fusion.final_emotion.upper()}**  \n"
        f"Valence: {fusion.valence:.2f} · Arousal: {fusion.arousal:.2f} · Confidence: {fusion.confidence:.2f}"
        )
        action_placeholder.info(suggest_action(fusion))
        # orb + label
        st.markdown(emotion_orb(fusion.valence, fusion.arousal), unsafe_allow_html=True)
        st.markdown(f"### {fusion.final_emotion.upper()} • Valence {fusion.valence:.2f} • Arousal {fusion.arousal:.2f}")

        # suggested micro-action card
        st.markdown(
            f"""
            <div class="emolens-card" style="background:#eef2ff;border-left:6px solid #4f46e5;">
              <b>💡 Suggested Action</b>
              <div class="small-muted" style="margin-top:6px;">{suggest_action(fusion)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Per-Modality Breakdown")
        # card wrapper
        st.markdown('<div class="emolens-card">', unsafe_allow_html=True)
        for name, m in fusion.modalities.items():
            st.write(f"**{name.upper()}**")
            st.write(vars(m))
        st.markdown('</div>', unsafe_allow_html=True)

        # LLM response (if any)
        if ai_reply:
            st.markdown("### 🤖 Adaptive AI Response")
            st.markdown(f"<div class='emolens-card'><pre style='white-space:pre-wrap'>{ai_reply}</pre></div>", unsafe_allow_html=True)

        if camera_bytes:
            st.image(camera_bytes, caption="Captured Frame", use_column_width=True)
    else:
        st.info("No fused emotional state yet. Provide camera/audio/text or type a short sentence.")

    st.markdown("---")
    if st.button("End Session & Save Timeline"):
        path = session_manager.end_session()
        st.success(f"Session saved → {path}")


###########################################
#   PAGE 2 — Educator Dashboard (Charts)   #
###########################################
elif page == "Educator Dashboard":
    st.header("Educator Dashboard — Sessions & Timeline")
    col_left, col_right = st.columns([1, 2])

    # Session Picker
    with col_left:
        st.subheader("Recent Sessions")
        sessions = fetch_sessions()
        if not sessions:
            st.info("No sessions found yet.")
            st.stop()

        selected_id = None
        for s in sessions:
            ts_short = (s["timestamp"][:19] if s.get("timestamp") else "")
            label = f"{s['session_id'][:8]}... — {ts_short}"
            if st.button(label, key=s['session_id']):
                selected_id = s['session_id']

        sid_list = [s["session_id"] for s in sessions]
        sid_choice = st.selectbox("Pick Session:", [""] + sid_list)
        if sid_choice:
            selected_id = sid_choice

    # Timeline & Metrics
    with col_right:
        if not selected_id:
            selected_id = sessions[0]['session_id']

        st.subheader(f"Session: {selected_id}")
        rows = fetch_session_rows(selected_id)
        if not rows:
            st.warning("No rows for this session.")
            st.stop()

        metrics = compute_session_metrics(rows)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Events", metrics["events"])
        c2.metric("Avg Valence", f"{metrics['avg_valence']:.2f}")
        c3.metric("Avg Arousal", f"{metrics['avg_arousal']:.2f}")
        c4.metric("Dominant Emotion", metrics["dominant_emotion"])

        # Build timeline df
        timestamps = []
        vals, aros, labels = [], [], []
        for r in rows:
            fe = r.get("fused_emotion") or {}
            ts = r.get("timestamp")
            timestamps.append(ts)
            vals.append(fe.get("valence", r.get("valence", 0)))
            aros.append(fe.get("arousal", r.get("arousal", 0)))
            labels.append(fe.get("emotion", r.get("emotion", "")))

        df = pd.DataFrame({"timestamp": timestamps, "valence": vals, "arousal": aros, "emotion": labels})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp")
        st.markdown("## 📊 Valence & Arousal Timeline")
        st.line_chart(df.set_index("timestamp")[["valence", "arousal"]])

        # Spikes
        spikes = detect_spikes(rows)
        if spikes:
            st.error(f"⚠️ Detected {len(spikes)} frustration spikes: {spikes}")
        else:
            st.success("No major spikes detected this session.")

        # Events table
        st.markdown("## 🔍 Events (latest → oldest)")
        display_rows = []
        for r in reversed(rows[-50:]):
            fused = r.get("fused_emotion") or {}
            display_rows.append({
                "ts": (r.get("timestamp") or "")[:19],
                "emotion": fused.get("emotion") or r.get("emotion"),
                "val": fused.get("valence") or r.get("valence"),
                "aro": fused.get("arousal") or r.get("arousal"),
                "action": r.get("brain_action") or ""
            })
        st.table(display_rows)

        # Export
        if st.button("Export JSON"):
            fname = f"export_{selected_id}.json"
            json.dump(rows, open(fname, "w"), indent=2, default=str)
            st.success(f"Saved to {fname}")

#########################
#   PAGE 3 — DB Test    #
#########################
elif page == "DB Test / Admin":
    st.header("DB Test / Admin Tools")
    st.write("Supabase URL:", os.getenv("SUPABASE_URL"))
    st.write("Key Exists:", bool(os.getenv("SUPABASE_KEY")))

    if st.button("Show Latest 10 Rows"):
        st.write(fetch_latest_rows(10))

    if st.button("Count Rows"):
        rows = fetch_latest_rows(5000)
        st.write(f"Total Rows in emotion_logs: {len(rows)}")

    st.markdown("### Local session files in root")
    for f in os.listdir("."):
        if f.startswith("session_") and f.endswith(".json"):
            st.write(f)


