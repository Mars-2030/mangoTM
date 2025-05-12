# utils.py
import re
from bs4 import BeautifulSoup
import emoji
import string
# We won't import nltk.corpus.stopwords directly here.
# Instead, the stopwords module will be passed as an argument to functions that need it.
# This makes this utils.py module less directly dependent on NLTK being fully set up
# at the moment of its import, deferring that dependency to the calling code.

# --- Text Preprocessing Functions ---

def remove_urls(text: str) -> str:
    """Removes URLs from a string."""
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_html(text: str) -> str:
    """Removes HTML tags from a string."""
    return BeautifulSoup(text, "html.parser").get_text()

def remove_special_chars(text: str) -> str:
    """
    Removes special characters, keeping alphanumeric, spaces, and basic 
    punctuation that might be part of words or sentiment.
    """
    return re.sub(r'[^A-Za-z0-9\s.,!?:;\'-@#_]', '', text)

def remove_hashtags_from_text(text: str) -> str: # Renamed to be more specific
    """Removes the # symbol and the tag text."""
    return re.sub(r'#\w+', '', text)

def extract_hashtags(text: str) -> list:
    """Extracts hashtags (including #) as a list from text."""
    return re.findall(r'#\w+', text)

def remove_mentions_from_text(text: str) -> str: # Renamed
    """Removes @ symbol and the mention text."""
    return re.sub(r'@\w+', '', text)

def extract_mentions(text: str) -> list:
    """Extracts mentions (including @) as a list from text."""
    return re.findall(r'@\w+', text)

def remove_emojis(text: str) -> str: # Renamed for clarity
    """Removes emojis from text."""
    return emoji.replace_emoji(text, replace='')

def demojize_emojis(text: str) -> str: # Renamed for clarity
    """Converts emojis to their text representation (e.g., :smile:)."""
    return emoji.demojize(text)

def convert_to_lowercase(text: str) -> str:
    """Converts text to lowercase."""
    return text.lower()

def remove_stopwords_from_text(text: str, nltk_stopwords_module, custom_stopwords: list = None) -> str:
    """
    Removes stopwords from tokenized text.
    `nltk_stopwords_module` should be the imported `nltk.corpus.stopwords` module.
    """
    if not nltk_stopwords_module:
        # This indicates an issue upstream (NLTK not ready or not passed)
        # A warning might be logged by the caller, or we can simply return the text.
        return str(text) 
    
    try:
        stop_words_set = set(nltk_stopwords_module.words('english'))
        if custom_stopwords:
            # Ensure custom_stopwords are lowercase for case-insensitive comparison
            custom_stopwords_lower = [stop.lower() for stop in custom_stopwords]
            stop_words_set.update(custom_stopwords_lower)
        
        words = str(text).split() # Simple whitespace tokenization
        filtered_words = [word for word in words if word.lower() not in stop_words_set]
        return " ".join(filtered_words)
    except Exception:
        # In case of any error (e.g., nltk_stopwords_module is not what's expected)
        # return the original text to prevent crashes.
        # The caller (e.g., in ui_components.py) can add more specific error handling/logging.
        return str(text)

def remove_numbers_from_text(text: str) -> str: # Renamed
    """Removes numbers from text."""
    return re.sub(r'\d+', '', text)

def remove_punctuation_from_text(text: str) -> str: # Renamed
    """
    Aggressively removes punctuation. 
    Note: If you need to extract hashtags/mentions first, run this after those operations.
    """
    # Keep @ and # if you intend to extract them before this step, 
    # otherwise, they will be removed by this.
    # The current version removes them.
    # To keep them: string.punctuation.replace('@','').replace('#','')
    return str(text).translate(str.maketrans('', '', string.punctuation))

# You could add more utility functions here as your project grows.
# For example, a function to handle date parsing consistently, etc.