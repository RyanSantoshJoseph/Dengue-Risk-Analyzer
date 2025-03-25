import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import requests
from gtts import gTTS  
import os  
from flask import Flask, request, jsonify

# Load Wav2Vec model for Speech-to-Text
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53")

# Flask app
app = Flask(__name__)

# 🔊 Convert Speech to Text (Google Wav2Vec)
def speech_to_text(audio_file):
    waveform, sample_rate = torchaudio.load(audio_file)
    input_values = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    return transcription

# 🦠 Get Dengue Prediction from Hugging Face Model
def get_dengue_prediction(symptoms_text):
    api_url = "https://ryanjoseph40-disease-prediction-of-dengue.hf.space/predict"
    data = {"query": symptoms_text}
    response = requests.post(api_url, json=data)
    return response.json()

# 🔊 Convert AI Response to Speech (Google TTS)
def text_to_speech(text, lang="hi"):  # Change "hi" for different regional languages
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save("response.mp3")
    os.system("start response.mp3")  # For Windows; use "afplay response.mp3" on macOS

# 📌 Flask API Endpoint for AI Assistant
@app.route('/voice-assistant', methods=['POST'])
def voice_assistant():
    audio_file = request.files['audio']
    
    # Step 1: Convert Speech to Text
    text_query = speech_to_text(audio_file)
    
    # Step 2: Get Prediction
    prediction_response = get_dengue_prediction(text_query)
    
    # Step 3: Convert Response to Speech
    ai_response = prediction_response.get("message", "Sorry, I could not process that.")
    text_to_speech(ai_response)
    
    return jsonify({"text": ai_response})

# Run Flask App
if __name__ == "__main__":
    app.run(debug=True)
