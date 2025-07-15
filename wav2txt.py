import whisper

WHISPER_MODEL_NAME = "large-v3"
model = whisper.load_model(WHISPER_MODEL_NAME)

def wav2txt(wav,language="en"):
    result = model.transcribe(wav,language=language)
    return result["text"]