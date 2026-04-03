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
    "ru": "Russian"
}

NLLB_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "sl": "slv_Latn",
    "ta": "tam_Taml",
    "de": "deu_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "zh": "zho_Hans",
    "ru": "rus_Cyrl"
}

# Detect source language
def detect_lang(text: str):
    try:
        return detect(text)
    except:
        return "en"
