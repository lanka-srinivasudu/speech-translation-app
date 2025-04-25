from deep_translator import GoogleTranslator

def split_text(text, max_chars=500):
    """
    Splits text into smaller chunks within the character limit for translation.
    """
    sentences = text.split(". ")  # Split by sentences
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def translate_text(text, target_lang="fr"):
    """
    Translates long text by splitting it into smaller chunks.
    """
    chunks = split_text(text)
    translated_chunks = [GoogleTranslator(source="auto", target=target_lang).translate(chunk) for chunk in chunks]
    return " ".join(translated_chunks)
