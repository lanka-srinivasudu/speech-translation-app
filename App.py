# import os
# import torch
# import torchaudio
# import speech_recognition as sr
# import whisper
# import streamlit as st
# from deep_translator import GoogleTranslator
# import streamlit as st
# from TTS.api import TTS
# import warnings
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
# from TTS.config.shared_configs import BaseDatasetConfig
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import torchaudio
# import asyncio
# import torch

# # Trial check
# import os
# os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# import torch

# # Import all custom configs that appear in the XTTS model checkpoint
# from TTS.tts.configs.xtts_config import XttsConfig
# from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
# from TTS.config.shared_configs import BaseDatasetConfig

# # Register them as safe globals for PyTorch unpickling
# torch.serialization.add_safe_globals([
#     XttsConfig,
#     XttsAudioConfig,
#     XttsArgs,
#     BaseDatasetConfig
# ])

# from TTS.api import TTS

# # Load XTTS model
# tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)

# #check ends

# warnings.filterwarnings("ignore")


# torch.serialization.add_safe_globals([XttsConfig])
# torch.serialization.add_safe_globals([XttsAudioConfig])
# torch.serialization.add_safe_globals([XttsArgs])
# torch.serialization.add_safe_globals([BaseDatasetConfig])

# LANGUAGE_MAP = {
#     "english": "en",
#     "spanish": "es",
#     "french": "fr",
#     "hindi": "hi",
#     "german": "de",
#     "italian": "it",
#     "korean": "ko"
# }


# try:
#     loop = asyncio.get_event_loop()
# except RuntimeError:
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)

# if not loop.is_running():
#     pass
# async def main():
#     pass

# if __name__ == "__main__":
#     asyncio.run(main())


# @st.cache_resource
# def load_models():
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
#     whisper_model = whisper.load_model("base").to(device)
#     return tts, whisper_model

# tts, whisper_model = load_models()

# if "initialized" not in st.session_state:
#     st.title("Speech Translation Integrated with Voice Cloning Technology")
#     st.session_state.initialized = True
#     st.success("Models loaded successfully!")
#     # st.success("Welcome to Speech Translation & Voice Cloning Technology!")



# AUDIO_FOLDER = 'audio_files'
# os.makedirs(AUDIO_FOLDER, exist_ok=True)


# def generate_audio(text, output_audio_path="output.wav", language="english"):
#     st.text("Audio generation Started....")
#     try:
#         if not text.strip():
#             st.error("Error: Empty text provided. Cannot generate audio.")
#             return None


#         lang_code = LANGUAGE_MAP.get(language.lower(), "en")  

        
#         tts.tts_to_file(text=text, speaker_wav="audio1.wav", language=lang_code, file_path=output_audio_path)

#         if not os.path.exists(output_audio_path):
#             st.error("Error: TTS model did not generate an audio file.")
#             return None
#         st.text(f"Audio generated, Saved as -> {output_audio_path}")
#         return output_audio_path
#     except Exception as e:
#         st.error(f"Error generating audio: {e}")
#         return None


# def split_text(text, max_chars=500):
#     sentences = text.split(". ")
#     chunks, current_chunk = [], ""
#     for sentence in sentences:
#         if len(current_chunk) + len(sentence) < max_chars:
#             current_chunk += sentence + ". "
#         else:
#             chunks.append(current_chunk.strip())
#             current_chunk = sentence + ". "
#     if current_chunk:
#         chunks.append(current_chunk.strip())
#     return chunks


# def translate_text(text, target_lang="english"):
#     try:
        
#         lang_code = LANGUAGE_MAP.get(target_lang.lower(), "en")  

#         chunks = split_text(text)
#         translated_chunks = [
#             GoogleTranslator(source="auto", target=lang_code).translate(chunk)
#             for chunk in chunks
#         ]
#         return " ".join(translated_chunks)
#     except Exception as e:
#         st.error(f"Error translating text: {e}")
#         return text  


# def transcribe_audio(audio_path):
#     try:
#         result = whisper_model.transcribe(audio_path, task="transcribe")
#         return result["text"], result["language"]
#     except Exception as e:
#         st.error(f"Error transcribing audio: {e}")
#         return "", "unknown"


# def generate_mel_spectrogram(audio_path):
#     try:
#         waveform, sample_rate = torchaudio.load(audio_path)
#         mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
#         return mel_spectrogram, sample_rate
#     except Exception as e:
#         st.error(f"Error generating Mel spectrogram: {e}")
#         return None, None


# def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
#     plt.figure(figsize=(10, 4))
    
    
#     mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()
    
    
#     plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='coolwarm')
#     plt.title(title)
#     plt.xlabel("Time")
#     plt.ylabel("Frequency")
#     plt.colorbar(format="%+2.0f dB")
#     plt.tight_layout()
#     st.pyplot(plt)


# lang = st.selectbox("Select Target Language", ["english", "spanish", "french", "hindi", "german", "italian", "korean"])


# if st.button("Start Transcription"):
#     with st.spinner("Listening for live audio..."):
#         with sr.Microphone() as source:
#             audio = recognizer.listen(source)
#             audio_path = os.path.join(AUDIO_FOLDER, "live_speech.wav")
#             with open(audio_path, "wb") as f:
#                 f.write(audio.get_wav_data())

    
#     st.text("Plotting Mel Spectrogram for Input Audio...")
#     mel_spectrogram, sample_rate = generate_mel_spectrogram(audio_path)
#     plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram of Input Audio")

    
#     detected_text, detected_lang = transcribe_audio(audio_path)
#     translated_text = translate_text(detected_text, target_lang=lang)

    
#     generated_audio_file = os.path.join(AUDIO_FOLDER, "live_output.wav")
#     audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)

#     if audio_path_generated:
#         st.audio(audio_path_generated, format="audio/wav")

        
#         st.text("Plotting Mel Spectrogram for Output Audio...")
#         mel_spectrogram_generated, _ = generate_mel_spectrogram(audio_path_generated)
#         plot_mel_spectrogram(mel_spectrogram_generated, title="Mel Spectrogram of Output Audio")

#         st.text(f"Detected: {detected_text}")
#         st.text(f"Translated: {translated_text}")
#     else:
#         st.error("Error generating audio")


# uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])
# if uploaded_file:
#     st.audio(uploaded_file, format="audio/wav")

    
#     file_path = os.path.join(AUDIO_FOLDER, "uploaded_audio.wav")
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

    
#     st.text("Plotting Mel Spectrogram for Uploaded Audio...")
#     mel_spectrogram, sample_rate = generate_mel_spectrogram(file_path)
#     plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram of Uploaded Audio")

    
#     detected_text, detected_lang = transcribe_audio(file_path)
#     st.text(f"Detected language: {detected_lang}")
#     st.text(f"Detected text: {detected_text}")
#     translated_text = translate_text(detected_text, target_lang=lang)
#     st.text(f"Translated text: {translated_text}")

#     generated_audio_file = os.path.join(AUDIO_FOLDER, "uploaded_output.wav")
#     audio_path_generated = generate_audio(translated_text, output_audio_path=generated_audio_file, language=lang)
    
#     st.audio(audio_path_generated, format="audio/wav")

#     st.text("Plotting Mel Spectrogram for Generated Audio ...")
#     mel_spectrogram, sample_rate = generate_mel_spectrogram(audio_path_generated)
#     plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram of Generated Audio")

######## Check check check triple
import os
import torch
import torchaudio
import whisper
import speech_recognition as sr
import streamlit as st
from deep_translator import GoogleTranslator
import matplotlib.pyplot as plt
import warnings

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Suppress file watcher errors (Streamlit Cloud)
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

# Trust custom XTTS classes for PyTorch 2.6+
torch.serialization.add_safe_globals([
    XttsConfig,
    XttsAudioConfig,
    XttsArgs,
    BaseDatasetConfig
])

# Hide warnings
warnings.filterwarnings("ignore")

# Language mapping
LANGUAGE_MAP = {
    "english": "en",
    "spanish": "es",
    "french": "fr",
    "hindi": "hi",
    "german": "de",
    "italian": "it",
    "korean": "ko"
}

# Load models with Streamlit cache
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False).to(device)
    whisper_model = whisper.load_model("base").to(device)
    return tts, whisper_model

tts, whisper_model = load_models()
recognizer = sr.Recognizer()

# UI Title
if "initialized" not in st.session_state:
    st.title("Speech Translation with Voice Cloning")
    st.session_state.initialized = True
    st.success("Models loaded successfully!")

# Audio folder
AUDIO_FOLDER = 'audio_files'
os.makedirs(AUDIO_FOLDER, exist_ok=True)

# Helper: Split long text for translation
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

# Translate
def translate_text(text, target_lang="english"):
    try:
        lang_code = LANGUAGE_MAP.get(target_lang.lower(), "en")
        chunks = split_text(text)
        translated_chunks = [GoogleTranslator(source="auto", target=lang_code).translate(chunk) for chunk in chunks]
        return " ".join(translated_chunks)
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Transcribe
def transcribe_audio(audio_path):
    try:
        result = whisper_model.transcribe(audio_path, task="transcribe")
        return result["text"], result["language"]
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return "", "unknown"

# Generate Mel Spectrogram
def generate_mel_spectrogram(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram()(waveform)
        return mel_spectrogram, sample_rate
    except Exception as e:
        st.error(f"Mel spectrogram error: {e}")
        return None, None

# Plot Spectrogram
def plot_mel_spectrogram(mel_spectrogram, title="Mel Spectrogram"):
    plt.figure(figsize=(10, 4))
    mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='coolwarm')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.colorbar(format="%+2.0f dB")
    plt.tight_layout()
    st.pyplot(plt)

# Generate audio
def generate_audio(text, output_audio_path="output.wav", language="english"):
    try:
        if not text.strip():
            st.error("Empty text. Cannot generate audio.")
            return None
        lang_code = LANGUAGE_MAP.get(language.lower(), "en")
        tts.tts_to_file(text=text, speaker_wav="audio1.wav", language=lang_code, file_path=output_audio_path)
        if not os.path.exists(output_audio_path):
            st.error("Audio file generation failed.")
            return None
        return output_audio_path
    except Exception as e:
        st.error(f"TTS error: {e}")
        return None

# Language selection
lang = st.selectbox("Select Target Language", list(LANGUAGE_MAP.keys()))

# Live Audio Recording
if st.button("Start Live Transcription"):
    with st.spinner("Recording..."):
        with sr.Microphone() as source:
            audio = recognizer.listen(source)
            audio_path = os.path.join(AUDIO_FOLDER, "live_input.wav")
            with open(audio_path, "wb") as f:
                f.write(audio.get_wav_data())

    st.text("Analyzing input audio...")
    mel_spectrogram, _ = generate_mel_spectrogram(audio_path)
    plot_mel_spectrogram(mel_spectrogram, "Input Audio Spectrogram")

    detected_text, detected_lang = transcribe_audio(audio_path)
    translated_text = translate_text(detected_text, target_lang=lang)

    output_path = os.path.join(AUDIO_FOLDER, "live_output.wav")
    audio_out = generate_audio(translated_text, output_audio_path=output_path, language=lang)

    if audio_out:
        st.audio(audio_out, format="audio/wav")
        mel_gen, _ = generate_mel_spectrogram(audio_out)
        plot_mel_spectrogram(mel_gen, "Output Audio Spectrogram")
        st.text(f"Detected: {detected_text}")
        st.text(f"Translated: {translated_text}")

# File Upload
uploaded_file = st.file_uploader("Upload Audio File", type=["wav"])
if uploaded_file:
    file_path = os.path.join(AUDIO_FOLDER, "uploaded_input.wav")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.audio(file_path, format="audio/wav")

    mel_spectrogram, _ = generate_mel_spectrogram(file_path)
    plot_mel_spectrogram(mel_spectrogram, "Uploaded Audio Spectrogram")

    detected_text, detected_lang = transcribe_audio(file_path)
    translated_text = translate_text(detected_text, target_lang=lang)

    output_path = os.path.join(AUDIO_FOLDER, "uploaded_output.wav")
    audio_out = generate_audio(translated_text, output_audio_path=output_path, language=lang)

    if audio_out:
        st.audio(audio_out, format="audio/wav")
        mel_gen, _ = generate_mel_spectrogram(audio_out)
        plot_mel_spectrogram(mel_gen, "Generated Output Audio")
        st.text(f"Detected: {detected_text}")
        st.text(f"Translated: {translated_text}")
