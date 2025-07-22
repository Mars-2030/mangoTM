import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.phrases import Phrases, Phraser
from gensim import matutils
from preprocessing import clean_and_tokenize, extract_hashtags, extract_mentions
import utils
import os

def gini(array):
    """Calculates the Gini coefficient of a numpy array."""
    if np.sum(array) == 0: 
        return 0.0
    array = np.array(array, dtype=np.float64)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def run_analysis(df, config):
    """
    Runs the full topic modeling pipeline using Gensim with optional phrase detection
    and extended evaluation metrics (C_v, U_Mass, Perplexity).
    """
    user_id_col = config['user_col']
    content_col = config['content_col']
    datetime_col = config['date_col']
    
    progress_bar = st.progress(0, text="Step 1/8: Validating data...")
    analysis_df = df.copy()
    analysis_df.dropna(subset=[user_id_col, content_col, datetime_col], inplace=True)
    analysis_df[datetime_col] = pd.to_datetime(analysis_df[datetime_col], errors='coerce')
    analysis_df.dropna(subset=[datetime_col], inplace=True)
    
    if config['cleaning_options'].get('hashtag_handling') == 'Extract (new column)':
        analysis_df['hashtags'] = analysis_df[content_col].astype(str).apply(extract_hashtags)
    if config['cleaning_options'].get('mention_handling') == 'Extract (new column)':
        analysis_df['mentions'] = analysis_df[content_col].astype(str).apply(extract_mentions)

    if analysis_df.empty:
        st.error("Error: No valid data remains after initial cleaning.")
        return None, None
    
    progress_bar.progress(12, text="Step 2/8: Tokenizing text...")
    chinese_stopwords = utils.load_chinese_stopwords() if config['language'] == 'Chinese' else None
    japanese_stopwords = utils.load_japanese_stopwords() if config['language'] == 'Japanese' else None
    
    analysis_df['tokenized_text'] = analysis_df[content_col].apply(
        lambda text: clean_and_tokenize(
            text, config['language'], config['cleaning_options'],
            chinese_stopwords=chinese_stopwords, japanese_stopwords=japanese_stopwords
        )
    )
    
    progress_bar.progress(24, text="Step 3/8: Filtering empty documents...")
    initial_doc_count = len(analysis_df)
    docs_to_model_df = analysis_df[analysis_df['tokenized_text'].map(len) > 0].copy()
    
    if docs_to_model_df.empty:
        st.error("Error: All documents are empty after cleaning. Please adjust cleaning options.")
        return None, None
        
    final_doc_count = len(docs_to_model_df)
    if initial_doc_count > final_doc_count:
        st.info(f"Filtered out {initial_doc_count - final_doc_count} of {initial_doc_count} documents that were empty after cleaning.")

    tokenized_docs = docs_to_model_df['tokenized_text'].tolist()
    
    if config.get('use_phrases', False):
        progress_bar.progress(36, text="Step 4/8: Detecting and combining phrases...")
        # FIX: Changed delimiter from b'_' (bytes) to '_' (string)
        phrases = Phrases(tokenized_docs, min_count=config['phrase_min_count'], threshold=config['phrase_threshold'], delimiter='_')
        bigram_phraser = Phraser(phrases)
        tokenized_docs = [bigram_phraser[doc] for doc in tokenized_docs]
        st.info("Phrase detection complete. Common bigrams are now treated as single tokens.")
    else:
        progress_bar.progress(36, text="Step 4/8: Skipping phrase detection...")

    progress_bar.progress(50, text="Step 5/8: Creating dictionary and corpus...")
    
    if config.get('use_tfidf', False):
        st.info("Applying TF-IDF for vocabulary selection...")
        docs_for_vectorizing = [' '.join(doc) for doc in tokenized_docs]
        tfidf_vectorizer = TfidfVectorizer(lowercase=False, token_pattern=r"(?u)\b\w[\w_]+\b")
        tfidf_matrix = tfidf_vectorizer.fit_transform(docs_for_vectorizing)
        mean_tfidf = tfidf_matrix.mean(axis=0).A1
        feature_names_tfidf = tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = pd.Series(mean_tfidf, index=feature_names_tfidf)
        top_n = config.get('top_n_tfidf', 2000)
        final_vocabulary_set = set(tfidf_scores.nlargest(top_n).index.tolist())
        
        tokenized_docs = [[token for token in doc if token in final_vocabulary_set] for doc in tokenized_docs]
        st.info(f"Using TF-IDF: Selected a vocabulary of {len(final_vocabulary_set)} terms.")
        
        non_empty_indices = [i for i, doc in enumerate(tokenized_docs) if doc]
        if len(non_empty_indices) < len(tokenized_docs):
            st.info(f"TF-IDF filtering removed {len(tokenized_docs) - len(non_empty_indices)} additional documents that became empty.")
            tokenized_docs = [tokenized_docs[i] for i in non_empty_indices]
            docs_to_model_df = docs_to_model_df.iloc[non_empty_indices].copy()
            
    dictionary = Dictionary(tokenized_docs)
    if not config.get('use_tfidf', False):
        dictionary.filter_extremes(no_below=config['no_below'], no_above=config['no_above'])
    if not dictionary:
        st.error("Error: No vocabulary remains after filtering. Please adjust your filtering parameters.")
        return None, None
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    if not corpus or all(not doc for doc in corpus):
        st.error("Error: The corpus is empty after dictionary creation. This is likely due to overly strict vocabulary filters.")
        return None, None
        
    progress_bar.progress(62, text="Step 6/8: Fitting Gensim LDA model...")
    workers = max(1, os.cpu_count() - 1) if os.cpu_count() else 1
    lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=config['num_topics'], random_state=42, passes=10, workers=workers)

    progress_bar.progress(75, text="Step 7/8: Calculating distributions and coherence...")
    doc_topic_dists_sparse = lda_model[corpus]
    doc_topic_dists = matutils.corpus2dense(doc_topic_dists_sparse, num_terms=config['num_topics']).T
    docs_to_model_df['dominant_topic'] = np.argmax(doc_topic_dists, axis=1)
    docs_to_model_df['topic_distribution'] = list(doc_topic_dists)
    analysis_df = analysis_df.merge(docs_to_model_df[['dominant_topic', 'topic_distribution']], left_index=True, right_index=True, how='left')
    analysis_df['topic_distribution'] = analysis_df['topic_distribution'].apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(config['num_topics']))

    try:
        coherence_model_cv = CoherenceModel(model=lda_model, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
        coherence_score_cv = coherence_model_cv.get_coherence()
    except Exception as e:
        st.warning(f"Could not calculate C_v coherence: {e}")
        coherence_score_cv = "N/A"
    
    try:
        coherence_model_umass = CoherenceModel(model=lda_model, corpus=corpus, dictionary=dictionary, coherence='u_mass')
        coherence_score_umass = coherence_model_umass.get_coherence()
    except Exception as e:
        st.warning(f"Could not calculate U_Mass coherence: {e}")
        coherence_score_umass = "N/A"
        
    try:
        perplexity_score = lda_model.log_perplexity(corpus)

        # log_perplexity = lda_model.log_perplexity(corpus)
        # perplexity_score = np.exp2(-log_perplexity)
    except Exception as e:
        st.warning(f"Could not calculate perplexity: {e}")
        perplexity_score = "N/A"

    progress_bar.progress(88, text="Step 8/8: Finalizing user analysis...")
    user_summary_gini = analysis_df.groupby(user_id_col)['topic_distribution'].apply(lambda dists: np.mean(np.vstack(dists), axis=0)).apply(gini).rename('gini_score')
    post_counts = analysis_df.groupby(user_id_col).size().rename('post_count')
    user_analysis = pd.concat([post_counts, user_summary_gini], axis=1).reset_index()

    class GensimLDAResults:
        def __init__(self, model, dictionary):
            self.model = model; self.dictionary = dictionary; self.components_ = model.get_topics()

    feature_names = [dictionary[i] for i in range(len(dictionary))]
    
    results = {
        'lda_model': GensimLDAResults(lda_model, dictionary),
        'feature_names': feature_names,
        'user_analysis': user_analysis, 
        'coherence_score_cv': coherence_score_cv,
        'coherence_score_umass': coherence_score_umass,
        'perplexity_score': perplexity_score,
        'config': config,
        'full_topic_distributions': analysis_df['topic_distribution'].tolist()
    }
    
    progress_bar.progress(100, text="Analysis Complete!")
    progress_bar.empty()
    return analysis_df, results