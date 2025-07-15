from openai import OpenAI
from app.config import settings

# ✅ Create Groq-compatible OpenAI client
client = OpenAI(
    api_key=settings.GROQ_API_KEY,
    base_url=settings.GROQ_URL,
)

class GroqAgent:
    def summarize_emotions(self, transcript):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Summarize the emotional journey of this call."},
                {"role": "user", "content": transcript}
            ]
        )
        return response.choices[0].message.content

    def extract_triggers(self, transcript):
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "Highlight emotionally significant statements or moments from this transcript."},
                {"role": "user", "content": transcript}
            ]
        )
        return response.choices[0].message.content
