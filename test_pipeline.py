# test_pipeline.py
from multimodal_emotion.video_emotion import analyze_video_frame
from multimodal_emotion.audio_emotion import analyze_audio
from multimodal_emotion.text_emotion import analyze_text
from multimodal_emotion.fusion import fuse

# simulate text
t = analyze_text("I am feeling tired and confused today.")

# simulate audio/video (None for now)
out = fuse(video=None, audio=None, text=t)

print("FINAL EMOTION STATE:\n", out)
