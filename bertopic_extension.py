## bertopic_extension.py

import pandas as pd
import numpy as np # Ensure numpy is imported
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import re
from collections import Counter
import json
from typing import List, Dict, Tuple, Union, Optional
import warnings
import os
from pathlib import Path

# NLP tools
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize # Keep word_tokenize

# Topic modeling with BERTopic
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from bertopic.vectorizers import ClassTfidfTransformer
    has_bertopic_deps = True
except ImportError as e:
    print(f"Warning: BERTopic dependencies missing ({e}). BERTopic functionality will be unavailable.")
    has_bertopic_deps = False
    # Define dummy classes or functions if needed elsewhere to avoid NameErrors
    class BERTopic: pass
    class SentenceTransformer: pass
    class UMAP: pass
    class HDBSCAN: pass


# For measuring inequality/concentration
from scipy.stats import entropy
import matplotlib.cm as cm
try:
    from wordcloud import WordCloud
    has_wordcloud = True
except ImportError:
    has_wordcloud = False

# --- Imports for MinHashLSH ---
try:
    from datasketch import MinHash, MinHashLSH
    has_datasketch = True
except ImportError:
    has_datasketch = False
# -----------------------------

# For Coherence Calculation
try:
    from gensim.models.coherencemodel import CoherenceModel
    from gensim import corpora
    has_gensim = True
except ImportError:
    print("Warning: Gensim not found. Coherence calculation will be skipped.")
    has_gensim = False


# Download necessary NLTK resources
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
# except LookupError:
#     print("Downloading NLTK data (punkt, stopwords)...")
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     print("NLTK data downloaded.")
os.environ['NLTK_DATA'] = ' /Users/mariamalmutairi/nltk_data' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class BERTopicModelingSystem:
    """
    A system for analyzing user post data using BERTopic.
    Discovers topic clusters, analyzes user topic diversity, and visualizes topic behavior over time.
    This class is designed to work with short texts where traditional LDA may struggle.
    """

    def __init__(self,
                 language_model: str = 'all-MiniLM-L6-v2',
                 time_bin: str = 'week',
                 min_topic_size: int = 10,
                 min_post_length: int = 3,
                 extra_stopwords: List[str] = None,
                 n_neighbors: int = 15,
                 n_components: int = 5,
                 min_cluster_size: int = 15,
                 num_topics: Optional[int] = None): # Changed to Optional[int]
        """
        Initialize the BERTopic modeling system.

        Args:
            language_model: Sentence transformer model to use
            time_bin: Time binning for temporal analysis ('day', 'week', 'month')
            min_topic_size: Minimum size of topics (used by BERTopic itself)
            min_post_length: Minimum number of words for a post to be considered after preprocessing
            extra_stopwords: Additional stopwords to remove during preprocessing
            n_neighbors: UMAP n_neighbors parameter
            n_components: UMAP n_components parameter
            min_cluster_size: HDBSCAN min_cluster_size parameter
            num_topics: Target number of topics to generate (if None or 'auto', determined automatically)
        """
        if not has_bertopic_deps:
             raise ImportError("BERTopic or its dependencies (sentence-transformers, umap-learn, hdbscan) are not installed. Cannot initialize BERTopicModelingSystem.")

        self.language_model_name = language_model # Store name
        self.embedding_model = None # Will be loaded later
        self.time_bin = time_bin
        self.min_topic_size = min_topic_size
        self.min_post_length = min_post_length
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        # Handle 'auto' or None for num_topics
        self.num_topics = num_topics if isinstance(num_topics, int) else None

        self.stop_words = set(stopwords.words('english'))
        if extra_stopwords: self.stop_words.update(extra_stopwords)

        # Instance variables initialization...
        self.data = None
        self.documents = None
        self.tokenized_documents = None
        self.doc_meta = None
        self.doc_timestamps = None
        self.topic_model = None
        self.doc_topics = None
        self.user_topic_distributions = None
        self.user_narrowness_scores = None
        self.temporal_topic_data = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """ Load data from CSV or JSONL file. """
        print(f"   Loading data from: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            data_list = []
            with open(file_path, 'r', encoding='utf-8') as f: # Added encoding
                for line in f:
                    try: data_list.append(json.loads(line))
                    except json.JSONDecodeError: print(f"   Warning: Skipping invalid JSON line: {line[:50]}...")
            df = pd.DataFrame(data_list)
        else: raise ValueError("Unsupported file format. Please provide CSV or JSONL.")
        required_cols = ['user_id', 'timestamp', 'post_content']; missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols: raise ValueError(f"Missing required columns: {missing_cols}")
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        initial_count = len(df); df = df.dropna(subset=['user_id', 'timestamp', 'post_content'])
        if len(df) < initial_count: print(f"   Dropped {initial_count - len(df)} rows due to missing essential data.")
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True)
        print(f"   Loaded {len(df)} valid records.")
        self.data = df
        return df

    def preprocess_text(self, text: str) -> str:
        """ Preprocess text for BERTopic (returns string). """
        if not isinstance(text, str): return ""
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
        text = re.sub(r'\d+', '', text) # Remove numbers
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra whitespace
        # Tokenize for stopword removal only, then rejoin
        tokens = word_tokenize(text)
        # Adjust min token length if needed (e.g., len(t) > 2)
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 1]
        processed_text = " ".join(tokens)
        return processed_text

    def preprocess_data(self, combine_by_window: bool = False) -> Tuple[List[str], List[List[str]], pd.DataFrame]:
        """ Preprocess posts, adds 'tokens' column to self.data. """
        if self.data is None: raise ValueError("No data loaded.")
        print("   Preprocessing text and creating tokens...")
        self.data['processed_text'] = self.data['post_content'].apply(self.preprocess_text)
        self.data['tokens'] = self.data['processed_text'].apply(word_tokenize)
        initial_count = len(self.data)
        # Filter based on word count in the *processed* text
        self.data = self.data[self.data['processed_text'].apply(lambda x: len(x.split())) >= self.min_post_length]
        if len(self.data) < initial_count: print(f"   Filtered out {initial_count - len(self.data)} posts shorter than {self.min_post_length} words.")
        if self.data.empty:
             print("   Error: No documents remaining after filtering by length."); self.documents = []; self.tokenized_documents = []; self.doc_meta = pd.DataFrame(); return [], [], pd.DataFrame()
        self.data = self.data.reset_index(drop=True)
        if combine_by_window:
            print(f"   Combining posts by user and time bin ({self.time_bin})...")
            df_for_combine = self.data[['user_id', 'timestamp', 'processed_text', 'tokens']].copy()
            if self.time_bin == 'day': df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.date
            elif self.time_bin == 'week': df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('W-MON').apply(lambda x: x.start_time.date())
            elif self.time_bin == 'month': df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('M').apply(lambda x: x.start_time.date())
            else: print(f"   Warning: Invalid time_bin '{self.time_bin}'. Defaulting to 'week'."); df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('W-MON').apply(lambda x: x.start_time.date())
            grouped = df_for_combine.groupby(['user_id', 'time_bin']); documents = []; tokenized_documents = []; doc_meta_list = []
            for idx, ((user_id, time_bin), group) in enumerate(grouped):
                combined_text = " ".join(group['processed_text']).strip()
                combined_tokens = [token for tokens_list in group['tokens'] for token in tokens_list]
                if combined_text and combined_tokens:
                    documents.append(combined_text); tokenized_documents.append(combined_tokens)
                    doc_meta_list.append({'user_id': user_id, 'time_bin': time_bin, 'timestamp': group['timestamp'].mean(), 'doc_id': idx })
            self.documents = documents; self.tokenized_documents = tokenized_documents; self.doc_meta = pd.DataFrame(doc_meta_list)
            self.doc_timestamps = self.doc_meta['timestamp'].tolist() if not self.doc_meta.empty else []
            print(f"   Created {len(self.documents)} combined documents for BERTopic.")
        else:
            print("   Using each post as a separate document for BERTopic.")
            self.documents = self.data['processed_text'].tolist(); self.tokenized_documents = self.data['tokens'].tolist()
            self.doc_meta = self.data[['user_id', 'timestamp']].copy(); self.doc_meta['doc_id'] = self.doc_meta.index
            self.doc_timestamps = self.data['timestamp'].tolist() # Use original post timestamps if not combining
        return self.documents, self.tokenized_documents, self.doc_meta

    def create_topic_model(self, documents: List[str], seed: int = 42) -> Optional[BERTopic]:
        """ Create and fit BERTopic model (Corrected: no timestamps in fit). """
        if not documents: print("   Error: Cannot create BERTopic model with no documents."); return None
        try:
            if self.embedding_model is None:
                print(f"   Loading language model: {self.language_model_name}")
                self.embedding_model = SentenceTransformer(self.language_model_name)
            print(f"   Initializing UMAP (n_neighbors={self.n_neighbors}, n_components={self.n_components})")
            umap_model = UMAP(n_neighbors=self.n_neighbors, n_components=self.n_components, min_dist=0.0, metric='cosine', random_state=seed, low_memory=True)
            print(f"   Initializing HDBSCAN (min_cluster_size={self.min_cluster_size})")
            hdbscan_model = HDBSCAN(min_cluster_size=self.min_cluster_size, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
            print("   Initializing CountVectorizer for topic representation")
            vectorizer_model = CountVectorizer(stop_words=list(self.stop_words))
            bertopic_params = {
                 "embedding_model": self.embedding_model, "umap_model": umap_model, "hdbscan_model": hdbscan_model,
                 "vectorizer_model": vectorizer_model, "ctfidf_model": ClassTfidfTransformer(reduce_frequent_words=True),
                 "min_topic_size": self.min_topic_size, "verbose": True, "calculate_probabilities": True
            }
            if self.num_topics is not None: print(f"   Setting target number of topics: {self.num_topics}"); bertopic_params["nr_topics"] = self.num_topics
            else: print("   Using automatic topic number detection.")
            topic_model = BERTopic(**bertopic_params)

            # --- CORRECTED FIT CALL ---
            print("   Fitting BERTopic model (this may take time)...")
            # Fit only with documents.
            self.topic_model = topic_model.fit(documents)
            # --- END CORRECTION ---

            # Print warning if timestamps were available but not used by fit
            if self.doc_timestamps and len(self.doc_timestamps) == len(documents):
                 print("   (Note: Timestamps available but not passed directly to BERTopic fit method.)")
            elif not self.doc_timestamps or len(self.doc_timestamps) != len(documents):
                 print("   Warning: Timestamps not available or length mismatch for documents.")

            # Get number of topics found (excluding outlier topic -1)
            num_found_topics = len([t for t in self.topic_model.get_topics() if t != -1])
            print(f"   BERTopic model fitting complete. Found {num_found_topics} topics (excluding outliers).")
            return self.topic_model
        except Exception as e:
            print(f"   Error during BERTopic model creation: {e}")
            import traceback; traceback.print_exc()
            self.topic_model = None; return None

    def get_document_topics(self) -> Optional[np.ndarray]:
        """ Get topic probability distribution for each document. """
        if self.topic_model is None: print("   Error: Topic model not created."); return None
        if not self.documents: print("   Error: No documents available."); return None
        print("   Calculating document topic probabilities...")
        try:
            # Check if probabilities were calculated during fit
            if hasattr(self.topic_model, 'probabilities_') and self.topic_model.probabilities_ is not None and len(self.topic_model.probabilities_) == len(self.documents):
                 print("   Using pre-calculated probabilities.")
                 topic_distr = self.topic_model.probabilities_
            else:
                 # Fallback to transform if probabilities missing or mismatched
                 print("   Calculating probabilities via transform (may take time)...")
                 if self.embedding_model is None: print("   Error: Embedding model not available for transform."); return None
                 # Check if embeddings are stored in the model object (common in newer BERTopic)
                 embeddings = None
                 if hasattr(self.topic_model, '_embeddings') and self.topic_model._embeddings is not None and len(self.topic_model._embeddings) == len(self.documents):
                      print("   Using existing embeddings from model object.")
                      embeddings = self.topic_model._embeddings
                 elif hasattr(self.topic_model, 'topic_embeddings_') and self.embedding_model: # Fallback to re-encoding
                     print("   Re-encoding documents to get embeddings...")
                     embeddings = self.embedding_model.encode(self.documents, show_progress_bar=True)

                 if embeddings is None:
                      print("   Error: Could not obtain document embeddings for transform.")
                      return None

                 # Use transform to get topics and probabilities
                 _, topic_distr = self.topic_model.transform(self.documents, embeddings)
                 print("   Probability calculation complete.")

            if topic_distr is not None:
                 self.doc_topics = np.array(topic_distr, dtype=np.float32)
                 print(f"   Calculated distributions for {len(self.doc_topics)} documents.")
                 return self.doc_topics
            else: print("   Error: Topic distribution calculation returned None."); self.doc_topics = None; return None
        except Exception as e: print(f"   Error calculating document topic distributions: {e}"); import traceback; traceback.print_exc(); self.doc_topics = None; return None

    def calculate_user_topic_distributions(self) -> Optional[Dict[str, np.ndarray]]:
        """ Calculate aggregated topic distribution for each user. """
        if self.doc_topics is None: 
            print("   Doc topics needed. Running...");
            if self.get_document_topics() is None: print("   Error: Failed doc topics."); return None
        if self.doc_meta is None or not all(c in self.doc_meta for c in ['user_id', 'doc_id']): print("   Error: doc_meta missing or invalid."); return None
        print("   Aggregating topic distributions per user...")
        user_topic_distributions = {} ; missing_doc_ids = 0
        if self.doc_meta.empty: print("   Warning: Document metadata is empty."); self.user_topic_distributions = {}; return {}
        # Use length of doc_meta as guide, assuming doc_topics matches after preprocessing
        if len(self.doc_topics) != len(self.doc_meta): print(f"   Warning: Mismatch doc_topics ({len(self.doc_topics)}) vs doc_meta ({len(self.doc_meta)}). Aggregation might be incorrect."); max_allowable_doc_id = min(len(self.doc_topics), len(self.doc_meta)) - 1
        else: max_allowable_doc_id = len(self.doc_topics) - 1
        for user_id in self.doc_meta['user_id'].unique():
            user_doc_ids = self.doc_meta[self.doc_meta['user_id'] == user_id]['doc_id'].values
            valid_doc_ids = [did for did in user_doc_ids if 0 <= did <= max_allowable_doc_id] # Ensure doc_id is valid index
            if len(valid_doc_ids) < len(user_doc_ids): missing_doc_ids += (len(user_doc_ids) - len(valid_doc_ids))
            if not valid_doc_ids: continue
            user_docs_topics = self.doc_topics[valid_doc_ids] # Index doc_topics with valid indices
            if user_docs_topics.size > 0 : user_topic_distributions[user_id] = user_docs_topics.mean(axis=0)
        if missing_doc_ids > 0: print(f"   Total missing/out-of-bounds doc_ids during user aggregation: {missing_doc_ids}")
        self.user_topic_distributions = user_topic_distributions; print(f"   Calculated distributions for {len(user_topic_distributions)} users."); return user_topic_distributions

    # --- Narrowness metric calculations (gini, entropy, top_n) ---
    def calculate_gini_coefficient(self, array: np.ndarray) -> float:
        """Calculate the Gini coefficient for a distribution."""
        if array is None or np.sum(array) == 0: return np.nan
        array = np.sort(np.abs(array))
        n = len(array) # Define n *after* sorting and getting the array length
        if n <= 1: return 0.0 # Now 'n' is defined before use
        index = np.arange(1, n + 1)
        # Gini coefficient formula
        denominator = n * np.sum(array) # Calculate denominator
        if denominator == 0: return np.nan # Avoid division by zero (handles edge case of all zeros after abs/sort)
        return (np.sum((2 * index - n - 1) * array)) / denominator

    def calculate_shannon_entropy(self, array: np.ndarray) -> float:
        """Calculate Shannon entropy for a distribution."""
        if array is None:
            return np.nan  # Handle None input

        # Ensure input is a flat numpy array
        array = np.asarray(array).flatten()

        # Filter out non-positive values to create clean_array
        clean_array = array[array > 0]

        # Check if the filtered array is empty
        if clean_array.size == 0:
            return 0.0 # Entropy of empty or all-zero distribution is 0

        # If not empty, normalize the filtered array (this prevents division by zero)
        array_sum = np.sum(clean_array)
        # Although clean_array.size > 0, sum could theoretically be zero if values are extremely small, though unlikely here. Add a check just in case.
        if array_sum == 0:
            return 0.0

        normalized_array = clean_array / array_sum

        # Calculate entropy using scipy.stats.entropy
        return entropy(normalized_array, base=2) # Use base 2 for bits
    
    def calculate_top_n_ratio(self, array: np.ndarray, n: int = 1) -> float:
        """Calculate the ratio of the top N topics in a distribution."""
        if array is None or n < 1:
            return np.nan # Handle invalid input

        # Ensure array is numpy array
        array = np.asarray(array)

        # Calculate the total sum *first*
        total_sum = np.sum(array)

        # Check if the total sum is zero (or very close to it)
        if total_sum < 1e-9: # Use tolerance for floating point
            return 0.0 # Ratio is 0 if total sum is 0

        # If sum is not zero, proceed with calculation
        sorted_array = np.sort(array)[::-1] # Sort descending
        top_n_sum = np.sum(sorted_array[:n])

        # Return the ratio
        return top_n_sum / total_sum
    
        # --- calculate_user_narrowness_scores ---
    def calculate_user_narrowness_scores(self) -> Optional[pd.DataFrame]:
        """ Calculate narrowness scores for each user. """
        if self.user_topic_distributions is None: 
            print("   User distributions needed. Running...");
            if self.calculate_user_topic_distributions() is None: print("   Error: Failed user distributions."); return None
        if self.doc_meta is None or self.doc_meta.empty: print("   Error: doc_meta required for post counts missing."); return None
        print("   Calculating user narrowness metrics..."); narrowness_data = []
        post_counts = self.doc_meta['user_id'].value_counts().to_dict()
        for user_id, topic_dist in self.user_topic_distributions.items():
            post_count = post_counts.get(user_id, 0); gini = self.calculate_gini_coefficient(topic_dist); entropy_score = self.calculate_shannon_entropy(topic_dist)
            top1 = self.calculate_top_n_ratio(topic_dist, n=1); top2 = self.calculate_top_n_ratio(topic_dist, n=2)
            dominant_topic_idx = np.argmax(topic_dist) if topic_dist.size > 0 else -1 # Index in probability array
            narrowness_data.append({'user_id': user_id, 'post_count': post_count, 'gini_coefficient': gini, 'shannon_entropy': entropy_score, 'top1_ratio': top1, 'top2_ratio': top2, 'dominant_topic': dominant_topic_idx})
        if not narrowness_data: print("   Warning: No narrowness data generated."); self.user_narrowness_scores = pd.DataFrame(columns=['user_id', 'post_count', 'gini_coefficient', 'shannon_entropy','top1_ratio', 'top2_ratio', 'dominant_topic']); return self.user_narrowness_scores
        narrowness_df = pd.DataFrame(narrowness_data).sort_values('gini_coefficient', ascending=False, na_position='last'); self.user_narrowness_scores = narrowness_df; print(f"   Calculated narrowness scores for {len(narrowness_df)} users."); return narrowness_df

    # --- calculate_temporal_topic_data ---
    def calculate_temporal_topic_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """ Calculate topic proportions over time for each user. """
        if self.doc_topics is None: 
            print("   Doc topics needed. Running...");
            if self.get_document_topics() is None: print("   Error: Failed doc topics."); return None
        if self.doc_meta is None or not all(c in self.doc_meta for c in ['user_id', 'timestamp', 'doc_id']): print("   Error: doc_meta missing or invalid."); return None
        # Check consistency again, using min length if mismatched
        if len(self.doc_topics) != len(self.doc_meta): print(f"   Warning: Mismatch doc_topics ({len(self.doc_topics)}) vs doc_meta ({len(self.doc_meta)})."); max_allowable_doc_id = min(len(self.doc_topics), len(self.doc_meta)) - 1
        else: max_allowable_doc_id = len(self.doc_topics) - 1
        print("   Calculating temporal topic evolution..."); temporal_data = {}; processed_users = 0
        if 'time_bin' in self.doc_meta.columns: time_col_in_meta = 'time_bin'; print(f"   Using pre-combined '{time_col_in_meta}'."); meta_df_for_temporal = self.doc_meta
        else:
            print(f"   Binning individual post timestamps using time_bin='{self.time_bin}'."); meta_df_for_temporal = self.doc_meta.copy(); meta_df_for_temporal['timestamp_dt'] = pd.to_datetime(meta_df_for_temporal['timestamp'], errors='coerce')
            meta_df_for_temporal = meta_df_for_temporal.dropna(subset=['timestamp_dt'])
            if self.time_bin == 'day': meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.date
            elif self.time_bin == 'week': meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('W-MON').apply(lambda p: p.start_time.date())
            elif self.time_bin == 'month': meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('M').apply(lambda p: p.start_time.date())
            else: print(f"   Warning: Invalid time_bin '{self.time_bin}'. Defaulting to 'week'."); meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('W-MON').apply(lambda p: p.start_time.date())
            time_col_in_meta = 'time_period'
        if meta_df_for_temporal.empty: print("   Warning: Metadata for temporal analysis is empty."); self.temporal_topic_data = {}; return {}
        grouped_by_user = meta_df_for_temporal.groupby('user_id'); num_topics_dist = self.doc_topics.shape[1]
        for user_id, user_meta_df in grouped_by_user:
            user_temporal = {}; grouped_by_time = user_meta_df.groupby(time_col_in_meta)
            for time_period, group_df in grouped_by_time:
                doc_ids_in_group = group_df['doc_id'].values; valid_doc_ids = [did for did in doc_ids_in_group if 0 <= did <= max_allowable_doc_id] # Use max allowable id
                if valid_doc_ids: 
                    bin_topic_dist = self.doc_topics[valid_doc_ids];
                    if bin_topic_dist.size > 0: 
                        user_temporal[time_period] = bin_topic_dist.mean(axis=0)
            if user_temporal:
                user_temporal_df = pd.DataFrame.from_dict(user_temporal, orient='index'); user_temporal_df.columns = [f'Topic_{i}' for i in range(num_topics_dist)] # Match columns to doc_topics shape
                user_temporal_df.index.name = 'time_bin'; user_temporal_df.index = pd.to_datetime(user_temporal_df.index); user_temporal_df = user_temporal_df.sort_index()
                temporal_data[user_id] = user_temporal_df; processed_users += 1
        self.temporal_topic_data = temporal_data; print(f"   Calculated temporal data for {processed_users} users."); return temporal_data

    # --- detect_suspicious_users (MinHashLSH Version) ---

    def detect_suspicious_users(self, post_frequency_threshold: float = 0.95,
                               narrowness_threshold: float = 0.95,
                               similarity_threshold: float = 0.8, # Jaccard threshold
                               minhash_permutations: int = 128) -> Optional[pd.DataFrame]:
        """ Flag suspicious users (MinHashLSH version). """
        if not has_datasketch:
             print("   Warning: 'datasketch' library not installed. Skipping duplicate detection.")
             # Fallback logic...
             if self.user_narrowness_scores is None:
                 if self.calculate_user_narrowness_scores() is None: return None
             post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
             valid_gini = self.user_narrowness_scores['gini_coefficient'].dropna()
             narrowness_cutoff = np.percentile(valid_gini, narrowness_threshold) if not valid_gini.empty else np.inf
             suspicious = self.user_narrowness_scores.copy()
             suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
             suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'].apply(lambda x: x >= narrowness_cutoff if pd.notna(x) else False)
             suspicious['duplicate_post_ratio'] = 0.0; suspicious['suspicious_duplicates'] = False
             suspicious['suspicious'] = suspicious['suspicious_post_freq'] & suspicious['suspicious_narrowness']
             suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], ascending=[False, False, False])
             print("   Finished flagging based on frequency and narrowness only.")
             return suspicious

        if self.user_narrowness_scores is None:
            if self.calculate_user_narrowness_scores() is None: print("   Error: Failed narrowness scores."); return None
        if self.data is None or 'tokens' not in self.data.columns:
             print("   Error: Precomputed 'tokens' column not found in self.data for duplicate check.");
             # Fallback logic...
             print("   Proceeding with suspicious detection based on frequency and narrowness only.")
             post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
             valid_gini = self.user_narrowness_scores['gini_coefficient'].dropna()
             narrowness_cutoff = np.percentile(valid_gini, narrowness_threshold) if not valid_gini.empty else np.inf
             suspicious = self.user_narrowness_scores.copy()
             suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
             suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'].apply(lambda x: x >= narrowness_cutoff if pd.notna(x) else False)
             suspicious['duplicate_post_ratio'] = 0.0; suspicious['suspicious_duplicates'] = False
             suspicious['suspicious'] = suspicious['suspicious_post_freq'] & suspicious['suspicious_narrowness']
             suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], ascending=[False, False, False])
             return suspicious

        print("   Calculating suspicious flags (frequency, narrowness)...")
        post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
        valid_gini = self.user_narrowness_scores['gini_coefficient'].dropna()
        narrowness_cutoff = np.percentile(valid_gini, narrowness_threshold) if not valid_gini.empty else np.inf
        suspicious = self.user_narrowness_scores.copy()
        suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
        suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'].apply(lambda x: x >= narrowness_cutoff if pd.notna(x) else False)

        print("   Calculating duplicate post ratios using MinHashLSH...")
        duplicate_post_ratios = {} ; processed_users_for_duplicates = 0
        users_to_check = suspicious['user_id'].unique()
        for user_id in users_to_check:
            user_rows = self.data[self.data['user_id'] == user_id];
            if user_rows.empty: duplicate_post_ratios[user_id] = 0.0; continue
            user_token_lists = user_rows['tokens'].tolist()
            user_token_lists = [t for t in user_token_lists if isinstance(t, list)] # Ensure they are lists
            if len(user_token_lists) <= 1: duplicate_post_ratios[user_id] = 0.0; continue
            minhashes = {}; token_sets = {}; post_keys = []
            for i, tokens in enumerate(user_token_lists):
                 original_index = user_rows.index[i]; key = f"post_{original_index}" # Use original index from self.data
                 if not tokens: continue
                 token_set = set(tokens)
                 if not token_set: continue
                 post_keys.append(key)
                 token_sets[key] = token_set
                 m = MinHash(num_perm=minhash_permutations)
                 for word in token_set: m.update(word.encode('utf8'))
                 minhashes[key] = m
            if len(minhashes) <= 1: duplicate_post_ratios[user_id] = 0.0; continue
            lsh = MinHashLSH(threshold=similarity_threshold, num_perm=minhash_permutations)
            indexed_keys = set(); [ (lsh.insert(k, m), indexed_keys.add(k)) for k, m in minhashes.items() ] # Index all minhashes
            similar_pairs_count = 0; checked_pairs = set()

            # --- CORRECTED LOOP SECTION ---
            for key in indexed_keys: # Query for each indexed key
                 # Ensure the key exists in minhashes before querying
                 if key not in minhashes:
                      continue

                 try:
                      # Assign query_minhash *inside* the try block, right before use
                      query_minhash = minhashes[key]
                      result = lsh.query(query_minhash)
                 except KeyError:
                      # This handles if the key IS in minhashes, but somehow NOT in LSH (should be rare)
                      print(f"   Warning: Key {key} not found in LSH index during query (User: {user_id}). Skipping.")
                      continue
                 except Exception as e: # Catch potential other errors during query
                      print(f"   Warning: Error querying LSH for key {key} (User: {user_id}): {e}")
                      continue

                 # --- Process results ---
                 for other_key in result:
                     if other_key not in indexed_keys or other_key not in token_sets: continue # Ensure candidate is valid
                     pair = tuple(sorted((key, other_key)))
                     if key == other_key or pair in checked_pairs: continue; checked_pairs.add(pair)
                     set1 = token_sets.get(key); set2 = token_sets.get(other_key)
                     if set1 is None or set2 is None: continue # Safety check
                     union_len = len(set1.union(set2))
                     if union_len == 0: actual_jaccard = 0.0
                     else: actual_jaccard = len(set1.intersection(set2)) / union_len
                     if actual_jaccard >= similarity_threshold: similar_pairs_count += 1 # Count verified pairs
            # --- END CORRECTED LOOP SECTION ---

            num_valid_posts = len(minhashes); total_possible_pairs = num_valid_posts * (num_valid_posts - 1) / 2 if num_valid_posts > 1 else 0
            duplicate_ratio = similar_pairs_count / max(1, total_possible_pairs); duplicate_post_ratios[user_id] = duplicate_ratio; processed_users_for_duplicates += 1
        print(f"   Calculated duplicate ratios for {processed_users_for_duplicates} users.")
        suspicious['duplicate_post_ratio'] = suspicious['user_id'].map(duplicate_post_ratios).fillna(0.0)
        suspicious['suspicious_duplicates'] = suspicious['duplicate_post_ratio'] > 0.5 # Threshold for flagging
        suspicious['suspicious'] = (suspicious['suspicious_post_freq'] & suspicious['suspicious_narrowness']) | suspicious['suspicious_duplicates']
        suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], ascending=[False, False, False], na_position='last')
        print(f"   Flagged {suspicious['suspicious'].sum()} users as suspicious."); return suspicious
    # --- Visualization methods ---
    def visualize_topic_words(self, n_words: int = 10, save_path: Optional[str] = None):
        """ Visualize top words for each BERTopic topic using word clouds. """
        if self.topic_model is None: print("   Error: BERTopic model not created."); return
        if not has_wordcloud: print("   WordCloud package not installed."); return
        print("   Generating BERTopic word clouds...")
        try:
            # Get topic info safely
            topic_info = self.topic_model.get_topic_info()
            if topic_info is None or topic_info.empty:
                print("   Warning: get_topic_info() returned empty or None.")
                return
            topic_info = topic_info[topic_info["Topic"] != -1].reset_index(drop=True)
        except Exception as e:
            print(f"   Error getting topic info from BERTopic model: {e}")
            return

        if topic_info.empty: print("   No valid topics found (excluding -1) for word clouds."); return
        n_topics_to_show = len(topic_info); n_cols = min(5, n_topics_to_show); n_rows = (n_topics_to_show + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(min(20, 5 * n_cols), 4 * n_rows), squeeze=False); axes = axes.flatten()
        shown_topics = 0

        for i, row in topic_info.iterrows():
            if i >= len(axes): break

            # --- Assign topic_id and topic_name BEFORE the try block ---
            topic_id = row["Topic"]
            # Use .get for safer access to potentially missing columns
            topic_name = row.get("CustomName", row.get("Name", f"Topic_{topic_id}"))
            # --- End Assignment Correction ---

            try:
                words_scores = self.topic_model.get_topic(topic_id)
                if not words_scores:
                    # print(f"   Warning: No words found for Topic {topic_id}. Skipping.") # Less verbose
                    axes[i].set_title(f'{topic_name}\n(No words)'); axes[i].axis('off'); continue

                top_words = {word: max(0.001, score) for word, score in words_scores[:n_words]}; # Ensure positive scores
                wc = WordCloud(width=400, height=300, background_color='white', max_words=n_words, colormap='viridis', prefer_horizontal=0.9);
                wc.generate_from_frequencies(top_words)
                axes[i].imshow(wc, interpolation='bilinear'); axes[i].set_title(f'{topic_name}'); axes[i].axis('off'); shown_topics += 1

            except Exception as e:
                # Now topic_id and topic_name are guaranteed to be defined here
                print(f"   Error generating word cloud for Topic {topic_id} ('{topic_name}'): {e}")
                axes[i].set_title(f'{topic_name}\n(Error)'); axes[i].axis('off')

        for j in range(shown_topics, len(axes)): axes[j].axis('off') # Hide unused
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        if save_path:
            try: plt.savefig(save_path, dpi=150); print(f"   BERTopic word clouds saved to {save_path}")
            except Exception as e: print(f"   Error saving BERTopic word clouds figure: {e}")
        plt.show(); plt.close(fig)

    def visualize_user_topic_pie(self, user_id: str, save_path: Optional[str] = None):
        """ Visualize topic distribution for a user as a pie chart. """
        if self.user_topic_distributions is None: print(f"   User distributions needed."); return
        if user_id not in self.user_topic_distributions: print(f"   User {user_id} not found."); return
        topic_dist = self.user_topic_distributions[user_id];
        if topic_dist is None or np.all(topic_dist < 1e-6): print(f"   User {user_id} has no significant distribution."); return
        threshold = 0.03; significant_indices = np.where(topic_dist > threshold)[0]; significant_probs = topic_dist[significant_indices]
        if len(significant_indices) == 0: 
            significant_indices = np.argsort(topic_dist)[-2:]; significant_probs = topic_dist[significant_indices]; non_zero_mask = significant_probs > 1e-6; significant_indices = significant_indices[non_zero_mask]; significant_probs = significant_probs[non_zero_mask];
            if len(significant_indices) == 0: print(f"   User {user_id} distribution is effectively zero."); return
        other_prob = 1.0 - np.sum(significant_probs); sort_order = np.argsort(significant_probs)[::-1]; significant_indices = significant_indices[sort_order]; significant_probs = significant_probs[sort_order]
        # Map indices to actual BERTopic IDs if needed (requires topic_info)
        # For now, using index as label
        labels = [f'Topic {i}' for i in significant_indices]; values = significant_probs.tolist();
        if other_prob > 0.01: labels.append('Other'); values.append(other_prob)
        fig, ax = plt.subplots(figsize=(8, 8)); colors = plt.cm.viridis(np.linspace(0, 1, len(values))) if len(values) > 20 else plt.cm.tab20.colors[:len(values)]
        wedges, texts, autotexts = ax.pie(values, autopct='%1.1f%%', startangle=90, colors=colors, pctdistance=0.85); [ (at.set_color('white'), at.set_fontsize(9), at.set_fontweight('bold')) for at in autotexts ]
        ax.axis('equal'); plt.title(f'Topic Distribution for User {user_id}', fontsize=14); ax.legend(wedges, labels, title="Topics (Indices)", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9) # Clarify legend uses index
        plt.tight_layout()
        if save_path: 
            try: plt.savefig(save_path, bbox_inches='tight')
            except Exception as e: print(f"   Error saving user pie chart {save_path}: {e}")
        plt.show(); plt.close(fig)

    def visualize_user_temporal_topics(self, user_id: str, save_path: Optional[str] = None):
        """ Visualize temporal topic distribution for a user. """
        if self.temporal_topic_data is None: print(f"   Temporal data needed."); return
        if user_id not in self.temporal_topic_data: print(f"   User {user_id} not found in temporal data."); return
        user_temporal_df = self.temporal_topic_data[user_id];
        if user_temporal_df.empty: print(f"   User {user_id} has no temporal data."); return
        num_topics_in_data = user_temporal_df.shape[1]; top_n = min(5, num_topics_in_data)
        if top_n <= 0: print(f"   No topic columns found for {user_id}."); return
        avg_topic_dist = user_temporal_df.mean().nlargest(top_n); top_topic_cols = avg_topic_dist.index.tolist(); plot_df = user_temporal_df.copy(); valid_top_cols = []
        for col in top_topic_cols:
            if col in plot_df: valid_top_cols.append(col)
            else: print(f"   Warning: Expected top topic column {col} not found for user {user_id}.")
        top_topic_cols = valid_top_cols; other_cols = [col for col in user_temporal_df.columns if col not in top_topic_cols]
        if other_cols: plot_df['Other'] = user_temporal_df[other_cols].sum(axis=1); plot_cols = top_topic_cols + ['Other']
        else: plot_cols = top_topic_cols
        if not plot_cols: print(f"   No columns to plot for user {user_id}."); return
        fig, ax = plt.subplots(figsize=(12, 6))
        try: 
            plot_df[plot_cols].plot.area(stacked=True, alpha=0.8, ax=ax, colormap='tab10' if len(plot_cols) <= 10 else 'viridis'); ax.set_title(f'Topic Evolution Over Time for User {user_id}', fontsize=14); ax.set_xlabel('Time', fontsize=12); ax.set_ylabel('Topic Proportion', fontsize=12); ax.legend(title='Topics (Indices)', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1) # Clarify legend
            if pd.api.types.is_datetime64_any_dtype(plot_df.index): fig.autofmt_xdate(); plt.tight_layout(rect=[0, 0, 0.85, 1])
            if save_path: 
                try: plt.savefig(save_path, bbox_inches='tight')
                except Exception as e: print(f"   Error saving user temporal plot {save_path}: {e}")
            plt.show()
        except Exception as plot_err: print(f"   Error generating temporal plot for user {user_id}: {plot_err}")
        finally: plt.close(fig)

    def visualize_narrowness_vs_frequency(self, metric: str = 'gini_coefficient', save_path: Optional[str] = None):
        """ Visualize user narrowness vs. post frequency. """
        if self.user_narrowness_scores is None or self.user_narrowness_scores.empty: print("   User narrowness scores unavailable."); return
        valid_metrics = ['gini_coefficient', 'shannon_entropy', 'top1_ratio', 'top2_ratio'];
        if metric not in valid_metrics: metric = 'gini_coefficient'; print(f"   Invalid metric, defaulting to '{metric}'.")
        if metric not in self.user_narrowness_scores.columns: print(f"   Metric column '{metric}' not found."); return
        plot_data = self.user_narrowness_scores[['user_id', 'post_count', metric]].dropna(subset=[metric, 'post_count']);
        if plot_data.empty: print(f"   No valid data points for metric '{metric}'."); return
        is_suspicious_col = 'suspicious' in self.user_narrowness_scores.columns;
        if is_suspicious_col: plot_data = plot_data.merge(self.user_narrowness_scores[['user_id', 'suspicious']], on='user_id', how='left'); plot_data['suspicious'] = plot_data['suspicious'].fillna(False)
        else: plot_data['suspicious'] = False
        print(f"   Visualizing {metric} vs Post Frequency..."); fig, ax = plt.subplots(figsize=(12, 8)); colors = ['red' if s else 'blue' for s in plot_data['suspicious']]; sizes = [50 if s else 20 for s in plot_data['suspicious']]; alpha = 0.6
        ax.scatter(plot_data['post_count'], plot_data[metric], c=colors, alpha=alpha, s=sizes, edgecolors='w', linewidths=0.5); title = f'User Topic Narrowness ({metric.replace("_", " ").title()}) vs. Post Frequency'
        if is_suspicious_col: title += f'\n({plot_data["suspicious"].sum()} Suspicious Users Highlighted)'
        ax.set_title(title, fontsize=14); ax.set_xlabel('Number of Posts (Log Scale)', fontsize=12); ax.set_ylabel(f'Topic Narrowness ({metric.replace("_", " ").title()})', fontsize=12); ax.grid(True, alpha=0.3); ax.set_xscale('log')
        if is_suspicious_col: from matplotlib.lines import Line2D; legend_elements = [Line2D([0], [0], marker='o', color='w', label='Not Suspicious', markerfacecolor='blue', markersize=8, alpha=alpha), Line2D([0], [0], marker='o', color='w', label='Suspicious', markerfacecolor='red', markersize=10, alpha=alpha)]; ax.legend(handles=legend_elements, loc='best', fontsize=9)
        plt.tight_layout();
        if save_path: 
            try: plt.savefig(save_path, dpi=150)
            except Exception as e: print(f"   Error saving narrowness plot: {e}")
        plt.show(); plt.close(fig)

    def visualize_topic_embedding(self, save_path: Optional[str] = None):
        """ Visualize topic embeddings using BERTopic's built-in visualization (HTML). """
        if self.topic_model is None: print("   Error: Topic model not created."); return
        print("   Generating BERTopic embedding visualization (interactive HTML)...")
        try: 
             topics_to_visualize = [tid for tid in self.topic_model.get_topics() if tid != -1];
             if not topics_to_visualize: print("   No topics found to visualize."); return
             # Check if visualize_topics method exists
             if not hasattr(self.topic_model, 'visualize_topics'): print("   Error: visualize_topics method not found in BERTopic model object."); return
             fig = self.topic_model.visualize_topics(topics=topics_to_visualize)
             if fig is None: print("   Warning: visualize_topics returned None."); return # Check return value
             if save_path:
                 save_path_html = Path(save_path)
                 if save_path_html.suffix.lower() != '.html': save_path_html = save_path_html.with_suffix('.html')
                 try: fig.write_html(str(save_path_html)); print(f"   BERTopic embedding visualization saved to {save_path_html}")
                 except Exception as e: print(f"   Error saving BERTopic HTML visualization: {e}")
             fig.show() # Attempt to show plot
        except AttributeError as ae: print(f"   Error accessing BERTopic visualization methods: {ae}")
        except Exception as e: print(f"   Error generating BERTopic embedding visualization: {e}")

    # --- Export methods ---
    def export_user_topic_data(self, output_path: str):
        """ Export user topic distribution and narrowness scores to CSV. """
        if self.user_narrowness_scores is None or self.user_narrowness_scores.empty: print("   User narrowness scores unavailable."); Path(output_path).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame().to_csv(output_path, index=False); return
        print(f"   Exporting user topic data to {output_path}..."); export_df = self.user_narrowness_scores.copy()
        if self.user_topic_distributions:
            first_user = next(iter(self.user_topic_distributions), None)
            if first_user and self.user_topic_distributions[first_user] is not None: # Check if dist is not None
                n_topics_dist = len(self.user_topic_distributions[first_user])
                topic_cols = {}; default_dist = np.zeros(n_topics_dist)
                for i in range(n_topics_dist): topic_col_name = f'Topic_{i}'; topic_cols[topic_col_name] = export_df['user_id'].map(lambda uid: self.user_topic_distributions.get(uid, default_dist)[i]) # Default handles missing users
                export_df = pd.concat([export_df, pd.DataFrame(topic_cols)], axis=1)
            else: print("   Warning: User topic distributions dict is empty or contains None.")
        else: print("   Warning: User topic distributions not available to add to export.")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True);
        try: export_df.to_csv(output_path, index=False, float_format='%.6f'); print(f"   Successfully exported user topic data.")
        except Exception as e: print(f"   Error exporting user topic data to {output_path}: {e}")

    def export_topic_data(self, output_dir: Union[str, Path]):
        """ Export BERTopic model data (info, words, embeddings). """
        if self.topic_model is None: print("   Error: Topic model not created."); return
        output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True); print(f"   Exporting BERTopic data to {output_dir}...")
        try:
            topic_info = self.topic_model.get_topic_info(); topic_info.to_csv(output_path / "topic_info.csv", index=False)
            topic_words = []; n_words_export = 20
            for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]: # Iterate through valid topics only
                 words_scores = self.topic_model.get_topic(topic_id)
                 if words_scores: # Check if topic has words
                     for i, (word, weight) in enumerate(words_scores[:n_words_export]): topic_words.append({'topic_id': topic_id, 'word': word, 'weight': weight, 'rank': i + 1})
            pd.DataFrame(topic_words).to_csv(output_path / "topic_words.csv", index=False, float_format='%.6f')
            if hasattr(self.topic_model, 'topic_embeddings_') and self.topic_model.topic_embeddings_ is not None: np.save(output_path / "topic_embeddings.npy", self.topic_model.topic_embeddings_)
            else: print("   Warning: Topic embeddings not found for export.")
            if self.temporal_topic_data: 
                import pickle;
                try:
                    with open(output_path / "temporal_topic_data.pkl", 'wb') as f: pickle.dump(self.temporal_topic_data, f)
                except Exception as e_pickle: print(f"   Error saving temporal data pickle: {e_pickle}")
            print(f"   Successfully exported BERTopic data.")
        except Exception as e: print(f"   Error during BERTopic data export: {e}")


    # --- run_full_pipeline ---
    def run_full_pipeline(self,
                         input_file: str,
                         output_dir: str,
                         combine_by_window: bool = True,
                         visualize_topics: bool = True,
                         detect_suspicious: bool = True):
        """ Run the complete BERTopic analysis pipeline. """
        start_time = datetime.now(); print(f"\n--- Starting BERTopic Pipeline: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Input file: {input_file}\nOutput dir: {output_dir}\nCombine by window: {combine_by_window}, Time bin: {self.time_bin}")
        output_path = Path(output_dir); output_path.mkdir(parents=True, exist_ok=True); results = {}
        try:
            print("\n1. Loading data..."); self.load_data(input_file)
            if self.data is None or self.data.empty: raise ValueError("Data loading failed.")
            results['initial_records'] = len(self.data)
            print("\n2. Preprocessing data..."); self.preprocess_data(combine_by_window=combine_by_window)
            if self.documents is None or not self.documents: raise ValueError("Preprocessing failed.")
            results['records_after_preprocessing'] = len(self.data); results['documents_for_bertopic'] = len(self.documents)
            print("\n3. Creating BERTopic model..."); self.create_topic_model(self.documents)
            if self.topic_model is None: raise ValueError("BERTopic model creation failed.")
            try: results['num_topics_found'] = len([t for t in self.topic_model.get_topics() if t != -1]) # Get actual count
            except Exception: results['num_topics_found'] = 'N/A'
            print("\n4. Calculating document topics...");
            if self.get_document_topics() is None: raise ValueError("Document topic calculation failed.")
            print("\n5. Calculating user topic distributions...");
            if self.calculate_user_topic_distributions() is None: print("Warning: Could not calculate user distributions.")
            results['num_users_processed'] = len(self.user_topic_distributions) if self.user_topic_distributions else 0
            print("\n6. Calculating user narrowness scores...");
            if self.calculate_user_narrowness_scores() is None: print("Warning: Could not calculate narrowness scores.")
            print("\n7. Calculating temporal topic data...");
            if self.calculate_temporal_topic_data() is None: print("Warning: Could not calculate temporal data.")

            # Coherence Calculation
            print("\n8. Calculating Topic Coherence..."); coherence_score = np.nan
            if not has_gensim: print("   Skipping coherence: Gensim not found.")
            else:
                try:
                    if self.topic_model and self.tokenized_documents:
                         valid_tokenized_docs = [doc for doc in self.tokenized_documents if doc]
                         if valid_tokenized_docs:
                             print(f"   Calculating c_v coherence on {len(valid_tokenized_docs)} tokenized documents...")
                             dictionary = corpora.Dictionary(valid_tokenized_docs)
                             if len(dictionary) > 0:
                                 topics_top_words = []; topic_ids = sorted([tid for tid in self.topic_model.get_topics() if tid != -1])
                                 for topic_id in topic_ids: 
                                    words = [word for word, score in self.topic_model.get_topic(topic_id)][:10];
                                    if words: 
                                        topics_top_words.append(words)
                                 if topics_top_words: coherence_model_bert = CoherenceModel(topics=topics_top_words, texts=valid_tokenized_docs, dictionary=dictionary, coherence='c_v'); coherence_score = coherence_model_bert.get_coherence(); print(f"   BERTopic Coherence Score (c_v): {coherence_score:.4f}"); results['coherence_score_c_v'] = coherence_score
                                 else: print("   Warning: No valid topic words found for coherence.")
                             else: print("   Warning: Gensim dictionary is empty.")
                         else: print("   Warning: No valid tokenized documents for coherence.")
                    else: print("   Warning: Model or tokenized docs unavailable for coherence.")
                    coherence_data = {'coherence_mean': coherence_score if pd.notna(coherence_score) else None}; coherence_path = output_path / "topic_coherence.json"
                    with open(coherence_path, 'w') as f: json.dump(coherence_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float_, np.float16, np.float32, np.float64)) else (None if pd.isna(x) else x))
                    print(f"   Coherence score saved to {coherence_path}")
                except Exception as e:
                    print(f"   Error calculating or saving coherence: {e}"); results['coherence_score_c_v'] = None;
                    try: 
                        coherence_data = {'coherence_mean': None}; coherence_path = output_path / "topic_coherence.json";
                        with open(coherence_path, 'w') as f: json.dump(coherence_data, f, indent=2)
                    except Exception as e_save: print(f"   Failed to save None coherence score either: {e_save}")

            step_num = 9; print(f"\n{step_num}. Exporting topic data..."); self.export_topic_data(output_path); step_num += 1
            if detect_suspicious:
                print(f"\n{step_num}. Detecting suspicious users..."); suspicious_users = self.detect_suspicious_users() # Uses MinHashLSH
                if suspicious_users is not None: suspicious_users.to_csv(output_path / "suspicious_users.csv", index=False); results['suspicious_users_detected'] = suspicious_users['suspicious'].sum(); print(f"   Identified {results['suspicious_users_detected']} suspicious users")
                else: print("   Warning: Suspicious user detection failed.")
                step_num += 1
            else: print(f"\n{step_num}. Skipping suspicious user detection."); step_num += 1
            print(f"\n{step_num}. Exporting user data..."); self.export_user_topic_data(output_path / "user_topic_data.csv"); step_num += 1
            if visualize_topics:
                print(f"\n{step_num}. Creating visualizations..."); step_num += 1
                self.visualize_topic_words(save_path=output_path / "topic_words.png")
                self.visualize_narrowness_vs_frequency(save_path=output_path / "narrowness_vs_frequency.png")
                self.visualize_topic_embedding(save_path=str(output_path / "topic_embedding.html")) # Ensure string path for plotly
                print("   Generating visualizations for top 5 users (by Gini)...")
                if self.user_narrowness_scores is not None and not self.user_narrowness_scores.empty:
                     top_users = self.user_narrowness_scores.head(5)['user_id'].astype(str).tolist()
                     for user_id in top_users: self.visualize_user_topic_pie(user_id, save_path=output_path / f"user_{user_id}_topics.png"); self.visualize_user_temporal_topics(user_id, save_path=output_path / f"user_{user_id}_temporal.png")
                else: print("   Skipping top user visualizations.")
            else: print(f"\n{step_num}. Skipping visualizations.")
        except Exception as pipeline_error: print(f"\n--- ERROR: BERTopic Pipeline failed ---\nError details: {pipeline_error}"); import traceback; traceback.print_exc(); print("---------------------------------------"); return None
        end_time = datetime.now(); print(f"\n--- BERTopic Pipeline finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---"); print(f"Total execution time: {end_time - start_time}\n----------------------------------------------------------")
        final_results = {'num_documents': results.get('documents_for_bertopic', 0), 'num_users': results.get('num_users_processed', 0), 'num_topics': results.get('num_topics_found', 0), 'output_dir': str(output_dir), 'coherence_score': results.get('coherence_score_c_v', None)}
        if detect_suspicious: final_results['suspicious_users_detected'] = results.get('suspicious_users_detected', 0)
        return final_results

# Example usage block (optional)
# if __name__ == "__main__":
#     # Initialize the system
#     topic_system = BERTopicModelingSystem(
#         language_model='all-MiniLM-L6-v2', # Fast, decent quality
#         time_bin='week',
#         min_topic_size=10,
#         min_post_length=5, # Require slightly longer posts for BERTopic
#         extra_stopwords=['rt', 'http', 'https', 'amp', 'via'],
#         n_neighbors=15,
#         n_components=5,
#         min_cluster_size=10,
#         num_topics=None # Let BERTopic decide number of topics initially
#     )
#
#     # Create dummy data if needed
#     # ...
#
#     # Run the pipeline
#     topic_system.run_full_pipeline(
#         input_file="preprocessed_data/streamlined_tweets_processed.csv", # Use appropriate preprocessed file
#         output_dir="topic_analysis_results/bertopic_results_test",
#         combine_by_window=False,
#         visualize_topics=True,
#         detect_suspicious=True
#     )