# app/utils.py
from langdetect import detect

# Map language codes to full names
LANG_MAP = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch"
}

# Detect source language
def detect_lang(text: str):
    try:
        return detect(text)
    except:
        return "en"
