from flask import Flask, render_template, request, jsonify
from src.stt.speech_to_text import SpeechToText
from src.tts.text_to_speech import TextToSpeech
from transformers import pipeline
import csv
import datetime

app = Flask(__name__)

# Initialize modules
stt = SpeechToText()
tts = TextToSpeech()
sentiment_analyzer = pipeline("sentiment-analysis")

# Log greetings
def log_greeting_response(response_text, sentiment):
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", "greeting_log.csv")
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write headers only if file is new
        if not file_exists:
            writer.writerow(["timestamp", "response_text", "sentiment_label", "sentiment_score"])
        writer.writerow([
            datetime.datetime.now().isoformat(),
            response_text,
            sentiment["label"],
            sentiment["score"]
        ])


# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Greeting / Voice input route
@app.route('/voice_input', methods=['POST'])
def voice_input():
    audio_file = request.files.get('audio')
    if not audio_file:
        return jsonify({"error": "No audio file received"}), 400
    print("✅ Audio received:", audio_file)

    candidate_response = stt.transcribe_audio_fileobj(audio_file)
    print("🗣️ Candidate response:", candidate_response)

    sentiment = sentiment_analyzer(candidate_response)[0]
    tts.speak(f"Tone: {sentiment['label']}")

    log_greeting_response(candidate_response, sentiment)

    return jsonify({
        "text": candidate_response,
        "sentiment": sentiment
    })



# Resume + JD analysis placeholder (future)
@app.route('/analyze_resume', methods=['POST'])
def analyze_resume():
    resume_file = request.files.get('resume')
    jd_file = request.files.get('jd')
    if not resume_file or not jd_file:
        return jsonify({"error": "Resume or JD missing"}), 400

    # Placeholder logic
    skill_match = {"match_percentage": 75, "skills_matched": ["Python", "ML"]}
    return jsonify(skill_match)

# Evaluation summary placeholder
@app.route('/summary')
def summary():
    # Placeholder for actual evaluation
    summary_data = {
        "strengths": ["Communication", "Problem-solving"],
        "weaknesses": ["Time management"],
        "confidence_score": 0.82,
        "recommendation": "Proceed to next round"
    }
    return render_template("summary.html", summary=summary_data)

if __name__ == "__main__":
    app.run(debug=True)
