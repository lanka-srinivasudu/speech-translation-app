import torch
import torchaudio
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig,XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig

# Register XttsConfig globally to allow loading
torch.serialization.add_safe_globals([XttsConfig])
torch.serialization.add_safe_globals([XttsAudioConfig])
torch.serialization.add_safe_globals([XttsArgs])
torch.serialization.add_safe_globals([BaseDatasetConfig])


# Set device (GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the FastPitch model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)

# Function to generate audio from text using the cloned voice
def generate_audio(text,output_audio_path="output.wav"):
    """
    Generate audio from input text by cloning the voice from the reference audio file.

    Args:
        text (str): Input text to be converted to speech.
        ref_audio_path (str): Path to the reference audio used for voice cloning.
        output_audio_path (str, optional): Path where the output audio is saved. Defaults to "output.wav".

    Returns:
        output_audio_path (str): Path where the output audio is saved.
    """
    # Step 1: Generate audio from the input text
    # This is a placeholder for your text-to-speech model's method
    tts.tts_to_file(text=text,speaker_wav="input1.wav",language="en",file_path=output_audio_path)

    # Step 2: Perform voice conversion to change the speaker's voice

    # Step 3: Return the output audio path
    return output_audio_path

