from flask import Flask, request, render_template
from pathlib import Path
from app.call_processor import CallProcessor
from app.agents.emotion_agent import EmotionAgent
from app.agents.groq_agent import GroqAgent
import os
import traceback

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load agents once
transcriber = CallProcessor()
emotion_agent = EmotionAgent()
groq_agent = GroqAgent()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            if "file" not in request.files:
                return render_template("index.html", error="No file part in request.")

            file = request.files["file"]
            if file.filename == "":
                return render_template("index.html", error="No file selected.")

            filepath = Path(UPLOAD_FOLDER) / file.filename
            file.save(filepath)

            # Process audio -> transcript
            transcript_data = transcriber.transcribe_call(filepath)
            print("📄 Transcript:", transcript_data["text"])

            # Emotion detection
            emotions = emotion_agent.get_emotions(transcript_data["text"])

            # Groq summarization and trigger detection
            summary = groq_agent.summarize_emotions(transcript_data["text"])
            triggers = groq_agent.extract_triggers(transcript_data["text"])

            return render_template("index.html",
                                   transcript=transcript_data["text"],
                                   language=transcript_data["language"],
                                   emotions=emotions,
                                   summary=summary,
                                   triggers=triggers)

        except Exception as e:
            print("🔥 Internal Error:", e)
            print(traceback.format_exc())
            return f"<h3>Internal Server Error</h3><pre>{traceback.format_exc()}</pre>", 500

    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True)
