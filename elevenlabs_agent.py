from elevenlabs.client import ElevenLabs
from app.config import settings

class ElevenLabsAgent:
    def __init__(self):
        self.client = ElevenLabs(
            api_key=settings.TextToSpeechElevenLabs
        )
        self.voice_id = settings.ELEVEN_VOICE_ID

    def generate_speech(self, text, output_path):
        audio = self.client.text_to_speech.convert(
            voice_id=self.voice_id,
            model_id="eleven_multilingual_v2",
            text=text,
            optimize_streaming_latency=0,
            output_format="mp3_44100_128"
        )

        # Convert generator into bytes
        audio_bytes = b"".join(audio)

        with open(output_path, "wb") as f:
            f.write(audio_bytes)
