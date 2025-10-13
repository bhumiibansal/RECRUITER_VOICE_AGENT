import os
import csv
from faster_whisper import WhisperModel

class SpeechToText:
    def __init__(self, model_size="small", device="cuda", compute_type="float16"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

  #      self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe_file(self, file_path: str) -> str:
        """Transcribe a single audio file and return text"""
        segments, _ = self.model.transcribe(file_path)
        text = " ".join([seg.text for seg in segments])
        return text

    def batch_transcribe(self, audio_folder: str, output_file: str):
        """Transcribe all audio files in a folder and save to CSV"""
        results = []

        for fname in os.listdir(audio_folder):
            if fname.endswith((".wav", ".mp3")):
                file_path = os.path.join(audio_folder, fname)
                try:
                    text = self.transcribe_file(file_path)
                    results.append([fname, text])
                    print(f"✅ Done: {fname}")
                except Exception as e:
                    print(f"❌ Error transcribing {fname}: {e}")

        # Save results to CSV
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "transcription"])
            writer.writerows(results)

        print(f"📄 All transcriptions saved at: {output_file}")
