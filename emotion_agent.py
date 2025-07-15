from transformers import pipeline

class EmotionAgent:
    def __init__(self):
        self.basic = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
        self.fine = pipeline("text-classification", model="bhadresh-savani/bert-base-go-emotion", top_k=None)

    def get_emotions(self, text):
        base = self.basic(text[:512])[0]
        fine = self.fine(text[:512])[0]
        return {
            "base_emotions": sorted(base, key=lambda x: x['score'], reverse=True),
            "go_emotions": sorted(fine, key=lambda x: x['score'], reverse=True)
        }