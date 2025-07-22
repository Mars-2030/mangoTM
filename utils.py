import streamlit as st
import os
import shutil
import tempfile
from pathlib import Path
import zipfile
import nltk

CHINESE_FONT_PATH = 'NotoSansSC-Regular.ttf'
JAPANESE_FONT_PATH = 'NotoSansJP-Regular.ttf'
CHINESE_STOPWORDS_PATH = 'cn_stopwords.txt'
JAPANESE_STOPWORDS_PATH = 'jp_stopwords.txt'

@st.cache_resource
def ensure_nltk_resources():
    """
    Ensures that necessary NLTK data packages are available, downloading them if not found.
    """
    try:
        app_nltk_data_path = Path(os.getcwd()) / "nltk_data_streamlit"
        app_nltk_data_path.mkdir(parents=True, exist_ok=True)
    except Exception:
        app_nltk_data_path = Path(tempfile.gettempdir()) / "streamlit_nltk_data"
        app_nltk_data_path.mkdir(parents=True, exist_ok=True)

    if str(app_nltk_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(app_nltk_data_path))

    resources_to_check = {
        "stopwords": ("corpora/stopwords", "stopwords"),
        "punkt": ("tokenizers/punkt", "punkt"),
        "wordnet": ("corpora/wordnet", "wordnet"),
        "averaged_perceptron_tagger": ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        "omw-1.4": ("corpora/omw-1.4", "omw-1.4"),
    }

    all_good = True
    for name, (path_suffix, package_id) in resources_to_check.items():
        try:
            nltk.data.find(path_suffix)
        except LookupError:
            try:
                st.info(f"Downloading NLTK resource: {name}...")
                nltk.download(package_id, download_dir=str(app_nltk_data_path), quiet=True)
                zip_file = app_nltk_data_path / f'{path_suffix}.zip'
                if zip_file.exists():
                    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                        zip_ref.extractall(os.path.dirname(zip_file))
                nltk.data.find(path_suffix)
            except Exception as e:
                st.error(f"Failed to download or verify NLTK resource '{name}': {e}")
                all_good = False
    return all_good

@st.cache_data
def load_chinese_stopwords():
    if not os.path.exists(CHINESE_STOPWORDS_PATH): return set()
    with open(CHINESE_STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

@st.cache_data
def load_japanese_stopwords():
    if not os.path.exists(JAPANESE_STOPWORDS_PATH): return set()
    with open(JAPANESE_STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}