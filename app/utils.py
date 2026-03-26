# app/utils.py
from langdetect import detect

# Map language codes to full names
LANG_MAP = {
    "en": "English",
    "fr": "French",
    "es": "Spanish",
    "ar": "Arabic",
    "sl": "Slovenian",
    "ta": "Tamil",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "zh": "Chinese",
    "ru": "Russsian"
}

# Detect source language
def detect_lang(text: str):
    try:
        return detect(text)
    except:
        return "en"
