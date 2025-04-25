import os
import torch
import torchaudio
import speech_recognition as sr
import whisper
from deep_translator import GoogleTranslator
import streamlit as st
from TTS.api import TTS
import warnings
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig


warnings.filterwarnings("ignore")

# Register XttsConfig globally to allow loading
torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([XttsArgs])
torch.serialization.add_safe_globals([BaseDatasetConfig])

# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models once to improve speed
try:
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    whisper_model = whisper.load_model("base")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

recognizer = sr.Recognizer()

# Directory for saving generated audio files
AUDIO_FOLDER = 'audio_files'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Function to generate audio from text
def generate_audio(text, output_audio_path="output.wav", language="en"):
    try:
        if not text.strip():
            st.error("Error: Empty text provided. Cannot generate audio.")
            return None

        # Generate TTS output
        tts.tts_to_file(text=text, speaker_wav="audio1.wav", language=language, file_path=output_audio_path)

        if not os.path.exists(output_audio_path):
            st.error("Error: TTS model did not generate an audio file.")
            return None
        return output_audio_path
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None

# Function to split text for translation
def split_text(text, max_chars=500):
    sentences = text.split(". ")
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# Function to translate text
def translate_text(text, target_lang="en"):
    try:
        chunks = split_text(text)
        translated_chunks = [
            GoogleTranslator(source="auto", target=target_lang).translate(chunk)
            for chunk in chunks
        ]
        return " ".join(translated_chunks)
    except Exception as e:
        st.error(f"Error translating text: {e}")
        return text  # Return original text if translation fails

# Function to transcribe speech to text
def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, task="transcribe")
        return result["text"], result["language"]
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return "", "unknown"

# Streamlit UI for transcribing
st.title("Real-Time Speech Translation & Cloning")

# Language selection dropdown
lang = st.selectbox("Select Target Language", ["en", "es", "fr", "hi"])

# Start transcription button
if st.button("Start Transcription"):
    with st.spinner("Listening for live audio..."):
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            audio_path = os.path.join(AUDIO_FOLDER, "live_speech.wav")
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())

    # Transcribing and translating the audio
    detected_text, detected_lang = transcribe_audio(audio_path)
    translated_text = translate_text(detected_text, target_lang=lang)

    # Generate audio from translated text
    generated_audio_file = os.path.join(AUDIO_FOLDER, "live_output.wav")
    audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)

    if audio_path_generated:
        st.audio(audio_path_generated, format="audio/wav")
        st.text(f"Detected: {detected_text}")
        st.text(f"Translated: {translated_text}")
    else:
        st.error("Error generating audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file
    file_path = os.path.join(AUDIO_FOLDER, "uploaded_audio.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Transcribe and translate uploaded audio
    detected_text, detected_lang = transcribe_audio(file_path)
    translated_text = translate_text(detected_text, target_lang=lang)

    # Generate audio from translated text
    generated_audio_file = os.path.join(AUDIO_FOLDER, "generated_audio.wav")
    audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)

    if audio_path_generated:
        st.audio(audio_path_generated, format="audio/wav")
        st.text(f"Detected: {detected_text}")
        st.text(f"Translated: {translated_text}")
    else:
        st.error("Error generating audio")
