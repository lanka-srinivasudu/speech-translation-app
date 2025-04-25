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
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Register XttsConfig globally to allow loading
torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([XttsArgs])
torch.serialization.add_safe_globals([BaseDatasetConfig])

# Load models once to improve speed
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    whisper_model = whisper.load_model("base").to(device)
    return tts, whisper_model

# Load models once
tts, whisper_model = load_models()

# Use session state to prevent reloads
if "initialized" not in st.session_state:
    st.title("Speech Translation Integrated with Voice Cloning Technology")
    st.session_state.initialized = True
    st.success("Models loaded successfully!")

recognizer = sr.Recognizer()

# Directory for saving generated audio files
AUDIO_FOLDER = 'audio_files'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Function to generate audio from text
def generate_audio(text, output_audio_path="output.wav", language="en"):
    st.text("Audio generation Started....")
    try:
        if not text.strip():
            st.error("Error: Empty text provided. Cannot generate audio.")
            return None

        # Generate TTS output
        tts.tts_to_file(text=text, speaker_wav="audio1.wav", language=language, file_path=output_audio_path)

        if not os.path.exists(output_audio_path):
            st.error("Error: TTS model did not generate an audio file.")
            return None
        st.text(f"Audio generated, Saved as -> {output_audio_path}")
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

# Function to generate mel spectrogram from audio
def generate_mel_spectrogram(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        return mel_spectrogram, sample_rate
    except Exception as e:
        st.error(f"Error generating Mel spectrogram: {e}")
        return None, None

# Function to plot mel spectrogram
def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    
    # Remove the batch dimension by squeezing it
    mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()
    
    # Plot the mel spectrogram
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(plt)


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

    # Plot mel spectrogram for input audio
    st.text("Plotting Mel Spectrogram for Input Audio...")
    mel_spectrogram, sample_rate = generate_mel_spectrogram(audio_path)
    plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram of Input Audio")

    # Transcribing and translating the audio
    detected_text, detected_lang = transcribe_audio(audio_path)
    translated_text = translate_text(detected_text, target_lang=lang)

    # Generate audio from translated text
    generated_audio_file = os.path.join(AUDIO_FOLDER, "live_output.wav")
    audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)

    if audio_path_generated:
        st.audio(audio_path_generated, format="audio/wav")

        # Plot mel spectrogram for generated audio
        st.text("Plotting Mel Spectrogram for Output Audio...")
        mel_spectrogram_generated, _ = generate_mel_spectrogram(audio_path_generated)
        plot_mel_spectrogram(mel_spectrogram_generated, sample_rate)

        st.text(f"Detected: {detected_text}")
        st.text(f"Translated: {translated_text}")
    else:
        st.error("Error generating audio")

# Upload audio file
uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")

    # Save the uploaded file
    file_path = os.path.join(AUDIO_FOLDER, "uploaded_audio.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Plot mel spectrogram for uploaded audio
    st.text("Plotting Mel Spectrogram for Uploaded Audio...")
    mel_spectrogram, sample_rate = generate_mel_spectrogram(file_path)
    plot_mel_spectrogram(mel_spectrogram, sample_rate)

    # Transcribe and translate uploaded audio
    detected_text, detected_lang = transcribe_audio(file_path)
    st.text(f"Detected language: {detected_lang}")
    st.text(f"Detected text: {detected_text}")
    translated_text = translate_text(detected_text, target_lang=lang)
    st.text(f"Translated text: {translated_text}")

    generated_audio_file = os.path.join(AUDIO_FOLDER, "uploaded_audio.wav")
    audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)
    
    st.audio(audio_path_generated, format="audio/wav")

    st.text("Plotting Mel Spectrogram for Generated Audio ...")
    mel_spectrogram, sample_rate = generate_mel_spectrogram(audio_path_generated)
    plot_mel_spectrogram(mel_spectrogram, sample_rate)


