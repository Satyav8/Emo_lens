
```markdown
# 🎭 EmoLens — Multimodal Emotion-Aware Learning Assistant  
### _Real-time Emotion Detection • Adaptive AI Teaching • Educator Dashboard • Multimodal Fusion_

EmoLens is an end-to-end **emotion-aware learning system** that analyzes **video**, **audio**, and **text** signals to understand a student’s emotional state in real time — and automatically adapts teaching responses using AI.

This system merges **AI emotional intelligence**, **adaptive learning**, **LLM reasoning**, and **live dashboards** into one cohesive platform.

---

## 🚀 Why EmoLens?

Traditional e-learning systems treat every student the same.  
EmoLens changes that.

By understanding how a student **feels** while learning, it can:

- Slow down when the student is frustrated  
- Offer motivation when valence is low  
- Increase difficulty when engagement is high  
- Modify teaching tone dynamically  
- Help educators view emotion timelines  

> 🧠 Imagine a tutor that *actually understands* student frustration, confusion, motivation, and focus — and reacts instantly.

---

# 🌟 Key Features

### 🎥 **Video Emotion Recognition (No TensorFlow + CPU friendly)**
Lightweight, heuristic-based face analysis using OpenCV:
- Smile recognition  
- Eye openness  
- Mouth contrast analysis  
- Face-region scoring  
- 7-emotion classification (happy, sad, angry, fear, disgust, surprise, neutral)

### 🔉 **Audio Emotion Classification**
Extracts MFCCs and predicts expressive states such as:
- Calm / Neutral  
- Happy  
- Sad  
- Angry  
- Fearful  
- Surprised  

### ✍️ **Text Sentiment & Emotion**
Uses a transformer-based NLP model to classify:
- Anger  
- Sadness  
- Joy  
- Neutral  
- Disgust  
- Surprise  
- Fear  

### 🔗 **Multimodal Fusion Engine**
Combines **video + audio + text** into a single **EmotionVector** containing:
- Final emotion  
- Valence (–1 to +1)  
- Arousal (0 to 1)  
- Confidence score  
- Per-modality breakdown  

### 🧠 **Adaptive Learning Brain**
A lightweight cognitive engine that maps emotions → intelligent micro-actions:
- “Slow down and clarify”  
- “Give motivational support”  
- “Increase challenge”  
- “Ask comprehension check question”  

### 🤖 **LLM Teaching Assistant**
Uses an OpenAI model to:
- Answer student questions  
- Adjust tone to student emotion  
- Provide step-by-step explanations  
- Maintain empathy + clarity  

### 📊 **Educator Dashboard**
View session summaries:
- Emotion timelines  
- Valence and arousal curves  
- Confusion/frustration spikes  
- Exportable session JSON files  
- Per-modality event table  

### 🗂️ **Supabase Integration**
Stores:
- Timestamped emotional events  
- Fused state  
- Brain recommendations  
- Session IDs  

Securely managed with Streamlit Secrets.

---

# 🧩 System Architecture

```

```
                 ┌────────────────────┐
                 │     Webcam Input    │
                 └──────────┬──────────┘
                            │
                            ▼
                 ┌────────────────────┐
                 │ Video Emotion Model │
                 └────────────────────┘

 ┌───────────────┐                    ┌──────────────────┐
 │  Audio Input   │                    │    Text Input     │
 └──────┬────────┘                    └─────────┬────────┘
        ▼                                          ▼
```

┌────────────────────┐                       ┌────────────────────┐
│ Audio Emotion Model │                       │ Text Emotion Model  │
└────────────────────┘                       └────────────────────┘

```
                ┌────────────────────────────┐
                │    Multimodal Fusion Engine │
                └──────────────┬─────────────┘
                               ▼
                ┌────────────────────────────┐
                │    adaptive_brain (AI)      │
                └──────────────┬─────────────┘
                               ▼
                ┌────────────────────────────┐
                │   LLM Teaching Response     │
                └────────────────────────────┘

                ┌────────────────────────────┐
                │  Educator Analytics Panel   │
                └────────────────────────────┘
```

````

---

# 🛠️ Tech Stack

### **Frontend**
- Streamlit  
- Custom UI components  
- Emotion Orb visualization  
- Real-time camera + audio input  

### **Backend**
- Python 3.13  
- Custom-built video emotion recognition  
- Audio MFCC + PyTorch classifier  
- Transformer-based text classification  
- Multimodal fusion logic  
- Adaptive Learning Brain  

### **Database**
- Supabase (PostgreSQL + Edge Functions)  

### **AI / NLP**
- OpenAI GPT model  
- Custom tone adaptation  

---

# 📦 Installation (Local)

```bash
git clone https://github.com/YOUR_USERNAME/EmoLens.git
cd EmoLens

# create venv
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
````

Create a `.env`:

```
SUPABASE_URL=...
SUPABASE_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o-mini
```

Run app:

```bash
streamlit run app.py
```

---

# ☁️ Deployment (Streamlit Cloud)

1. Push project to GitHub
2. Open [https://share.streamlit.io](https://share.streamlit.io)
3. New App → Select Repo → Choose `app.py`
4. Add Secrets:

```
SUPABASE_URL="..."
SUPABASE_KEY="..."
OPENAI_API_KEY="..."
OPENAI_MODEL="gpt-4o-mini"
```

5. Deploy 🎉

* Multimodal emotion detection
* Adaptive learning responses
* Supabase session tracking
* Dashboard & analytics
* AI conversational layer

### 🔵 In Progress

* Higher-accuracy FER region cropping
* Improved audio emotion stability
* Enhanced UI theme

### 🟣 Future Enhancements

* YOLO-based face recognition
* Attention tracking (gaze detection)
* Classroom multi-user mode
* Instructor analytics (weekly summaries)
* Personalization model per student

---

# 💡 Inspiration

EmoLens was built to explore a future where:

**AI understands not just what students know —
but how they feel while learning.**

This project demonstrates the power of combining:

* emotional intelligence
* personalized education
* multimodal AI
* real-time analytics

---

# 🤝 Contributing

Contributions, issues, and feature requests are welcome!

If you’d like to add a new modality, improve accuracy, or extend the dashboard — feel free to open a PR.

---

# ✨ Author

**Satya Prabhas** (2025)
Developer • Innovator • AI Enthusiast


