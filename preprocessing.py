import pandas as pd
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
import jieba
from janome.tokenizer import Tokenizer

_janome_tokenizer = None
_wordnet_lemmatizer = None
_stopwords_cache = {}

def _get_janome_tokenizer():
    global _janome_tokenizer
    if _janome_tokenizer is None:
        _janome_tokenizer = Tokenizer()
    return _janome_tokenizer

def _get_wordnet_lemmatizer():
    global _wordnet_lemmatizer
    if _wordnet_lemmatizer is None:
        _wordnet_lemmatizer = WordNetLemmatizer()
    return _wordnet_lemmatizer

def _get_stopwords(language):
    if language not in _stopwords_cache:
        try:
            if language == 'english':
                _stopwords_cache[language] = set(stopwords.words('english'))
            elif language == 'spanish':
                _stopwords_cache[language] = set(stopwords.words('spanish'))
            else:
                _stopwords_cache[language] = set()
        except Exception:
            _stopwords_cache[language] = set()
    return _stopwords_cache[language]

def extract_hashtags(text):
    if pd.isna(text): return []
    try:
        return re.findall(r"#([^\s#@]+)", str(text), flags=re.UNICODE)
    except Exception:
        return []

def extract_mentions(text):
    if pd.isna(text): return []
    try:
        return re.findall(r"@([^\s#@]+)", str(text), flags=re.UNICODE)
    except Exception:
        return []

def clean_and_tokenize(text, lang, options, chinese_stopwords=None, japanese_stopwords=None):
    if pd.isna(text): return []
    text = str(text)
    if not text.strip(): return []

    if options.get('lowercase', True): text = text.lower()
    if options.get('remove_html', True): text = re.sub(r'<[^<>]+>', '', text)
    if options.get('remove_urls', True):
        url_patterns = [r'https?://[^\s<>"{}|\\^`\[\]]+', r'www\.[^\s<>"{}|\\^`\[\]]+']
        for pattern in url_patterns:
            text = re.sub(pattern, ' ', text, flags=re.IGNORECASE)
    
    if options.get('mention_handling') == 'Remove': text = re.sub(r'@[^\s#@]+', ' ', text, flags=re.UNICODE)
    if options.get('hashtag_handling') == 'Remove': text = re.sub(r'#[^\s#@]+', ' ', text, flags=re.UNICODE)

    if options.get('emoji_handling') == 'Remove Emojis': text = emoji.replace_emoji(text, replace=' ')
    elif options.get('emoji_handling') == 'Convert Emojis to Text': text = emoji.demojize(text, delimiters=(" ", " "))

    if lang in ['Chinese', 'Japanese']:
        if options.get('remove_special_chars', False) or options.get('remove_punctuation', False):
            cjk_pattern = r'[^\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7afa-zA-Z0-9\s]'
            text = re.sub(cjk_pattern, ' ', text, flags=re.UNICODE)
    else:
        if options.get('remove_punctuation', False): text = re.sub(r'[^\w\s]', ' ', text)
        if options.get('remove_special_chars', False): text = re.sub(r'[^a-zA-Z0-9\s.,!?-]', ' ', text)

    if options.get('remove_numbers', False): text = re.sub(r'\d+', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    if not text: return []

    try:
        if lang == 'Chinese': tokens = jieba.lcut(text)
        elif lang == 'Japanese': tokens = [token.surface for token in _get_janome_tokenizer().tokenize(text) if token.surface.strip()]
        else: tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()

    tokens = [token.strip() for token in tokens if token.strip()]
    if not tokens: return []

    if lang == 'English' and options.get('lemmatize', False):
        lemmatizer = _get_wordnet_lemmatizer()
        try:
            tagged_tokens = pos_tag(tokens)
            lemmatized_tokens = []
            for token, tag in tagged_tokens:
                pos = 'n'
                if tag.startswith('J'): pos = 'a'
                elif tag.startswith('V'): pos = 'v'
                elif tag.startswith('R'): pos = 'r'
                lemmatized_tokens.append(lemmatizer.lemmatize(token, pos))
            tokens = lemmatized_tokens
        except Exception:
            tokens = [lemmatizer.lemmatize(token) for token in tokens]

    if options.get('remove_stopwords', False):
        stop_words = set()
        if lang == 'English': stop_words.update(_get_stopwords('english'))
        elif lang == 'Spanish': stop_words.update(_get_stopwords('spanish'))
        if lang == 'Chinese' and chinese_stopwords: stop_words.update(chinese_stopwords)
        elif lang == 'Japanese' and japanese_stopwords: stop_words.update(japanese_stopwords)
        if options.get('custom_stopwords'):
            stop_words.update([word.strip().lower() for word in options['custom_stopwords'].split(',') if word.strip()])
        
        tokens = [token for token in tokens if token not in stop_words]

    min_len = options.get('min_token_length', 1)
    if min_len > 1:
        tokens = [token for token in tokens if len(token) >= min_len]

    return tokens