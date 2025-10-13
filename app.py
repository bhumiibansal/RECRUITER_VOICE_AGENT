from src.stt.speech_to_text import SpeechToText

if __name__ == "__main__":
    audio_folder = "data/clips"
    output_file = "data/transcriptions.csv"

    stt = SpeechToText()
    stt.batch_transcribe(audio_folder, output_file)
    # app.py
print("🚀 Starting Recruiter Voice Agent tests...")

# 1. Speech-to-Text
print("✅ speech_to_text imported")
