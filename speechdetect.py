import speech_recognition as sr
import whisper
from speechtranslator import translate_text
from voicecloning import generate_audio
import warnings
warnings.filterwarnings("ignore")


def transcribe_audio(audio_path):
    """
    Converts speech from an audio file to text using Whisper.
    """
    model = whisper.load_model("large")  # Use "large" for better accuracy
    result = model.transcribe(audio_path,task="transcribe")
    lang=result["language"]
    print(f"Detected Language is : {lang}")
    return result["text"]

r = sr.Recognizer()
while True:
        try:
            with sr.Microphone() as source:
                print("Listening... (say 'exit' to quit)")
                audio = r.listen(source)
                detected_text = transcribe_audio(audio)
                print(f"🎙️ Detected Speech: {detected_text}")
                #
                if speech_text.lower() == 'exit':  
                    print("Exiting the translation.")
                    break

                # Translate speech to Hindi
                translated_text=translate_text(detected_text, target_lang="hi")
                generated_audio_path=generate_audio(translated_text)


        except sr.UnknownValueError:
            print("Couldn't understand the speech. Please try again.")
        except sr.RequestError:
            print("Couldn't process the request from Google. Check your internet connection.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

