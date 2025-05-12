# user_topic_modeling.py

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
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Topic modeling
import gensim
from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel # Added for coherence

# Dimensionality reduction for visualization
try:
    import umap
    from sklearn.manifold import TSNE
    has_dim_reduction = True
except ImportError:
    has_dim_reduction = False

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

# Download necessary NLTK resources
# try:
#     nltk.data.find('tokenizers/punkt')
#     nltk.data.find('corpora/stopwords')
#     nltk.data.find('corpora/wordnet')
# except LookupError:
#     print("Downloading NLTK data (punkt, stopwords, wordnet)...")
#     nltk.download('punkt', quiet=True)
#     nltk.download('stopwords', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     print("NLTK data downloaded.")
os.environ['NLTK_DATA'] = ' /Users/mariamalmutairi/nltk_data' 
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class UserTopicModelingSystem:
    """
    A system for analyzing user post data using LDA topic modeling.
    Discovers topic clusters, analyzes user topic diversity, and visualizes topic behavior over time.
    """

    def __init__(self,
                 num_topics: int = 20,
                 time_bin: str = 'week',
                 lemmatize: bool = True,
                 min_post_length: int = 3,
                 extra_stopwords: List[str] = None):
        """
        Initialize the topic modeling system.

        Args:
            num_topics: Number of topics for LDA model
            time_bin: Time binning for temporal analysis ('day', 'week', 'month')
            lemmatize: Whether to lemmatize tokens
            min_post_length: Minimum number of tokens for a post to be considered
            extra_stopwords: Additional stopwords to remove
        """
        self.num_topics = num_topics
        self.time_bin = time_bin
        self.lemmatize = lemmatize
        self.min_post_length = min_post_length

        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        if extra_stopwords:
            self.stop_words.update(extra_stopwords)

        if lemmatize:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.stemmer = PorterStemmer()

        # Will be populated after processing
        self.data = None # Should contain raw data + 'tokens' after preprocessing
        self.documents = None # List of token lists used for LDA model building
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
        self.doc_meta = None # DataFrame mapping doc_id to user, time etc.
        self.doc_topics = None
        self.user_topic_distributions = None
        self.user_narrowness_scores = None
        self.temporal_topic_data = None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or JSONL file.

        Args:
            file_path: Path to input file

        Returns:
            DataFrame with the loaded data
        """
        print(f"   Loading data from: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            data_list = [] # Renamed to avoid conflict
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        data_list.append(json.loads(line))
                    except json.JSONDecodeError:
                         print(f"   Warning: Skipping invalid JSON line: {line[:50]}...")
            df = pd.DataFrame(data_list)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSONL.")

        # Ensure required columns exist
        required_cols = ['user_id', 'timestamp', 'post_content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

        # Drop rows with conversion errors or missing essential data
        initial_count = len(df)
        df = df.dropna(subset=['user_id', 'timestamp', 'post_content'])
        if len(df) < initial_count:
             print(f"   Dropped {initial_count - len(df)} rows due to missing essential data (user_id, timestamp, content, or invalid timestamp).")


        # Sort by user_id and timestamp
        df = df.sort_values(['user_id', 'timestamp']).reset_index(drop=True) # Reset index

        print(f"   Loaded {len(df)} valid records.")
        self.data = df # Store the loaded data
        return df

    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: lowercase, remove stopwords, URLs, emojis, and tokenize.
        This version is specific to LDA, returning a list of tokens.

        Args:
            text: Input text to preprocess

        Returns:
            List of preprocessed tokens
        """
        if not isinstance(text, str):
            return []

        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)

        # Remove user mentions (optional, common in tweets)
        text = re.sub(r'@\w+', '', text)

        # Remove hashtags symbol but keep the text (optional)
        # text = text.replace('#', '')

        # Remove punctuation and numbers (adjust if numbers are important)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]

        # Lemmatize or stem
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        else:
            tokens = [self.stemmer.stem(t) for t in tokens]

        return tokens

    def preprocess_data(self, combine_by_window: bool = False) -> List[List[str]]:
        """
        Preprocess all posts in the dataset for LDA.
        Also adds a 'tokens' column to self.data for later use (e.g., suspicious detection).

        Args:
            combine_by_window: Whether to combine posts into time windows per user

        Returns:
            List of preprocessed documents (each document is a list of tokens) for LDA model.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        print("   Preprocessing text and creating tokens...")
        # --- Add 'tokens' column to self.data ---
        # This preprocesses each original post individually
        self.data['tokens'] = self.data['post_content'].apply(self.preprocess_text)
        # Filter out posts with too few tokens
        self.data = self.data[self.data['tokens'].apply(len) >= self.min_post_length]
        self.data = self.data.reset_index(drop=True)

        # --- Build and apply bigram model ---
        from gensim.models.phrases import Phrases, Phraser

        # Only if documents exist
        if not self.data.empty:
            sentences = self.data['tokens'].tolist()
            bigram = Phrases(sentences, min_count=5, threshold=10)
            bigram_mod = Phraser(bigram)

            # Transform tokens
            self.data['tokens'] = self.data['tokens'].apply(lambda tokens: bigram_mod[tokens])
        # -------------------------------------
        # -----------------------------------------

        # Filter out posts that are too short based on tokens
        initial_count = len(self.data)
        self.data = self.data[self.data['tokens'].apply(len) >= self.min_post_length]
        if len(self.data) < initial_count:
            print(f"   Filtered out {initial_count - len(self.data)} posts shorter than {self.min_post_length} tokens.")

        # Reset index after filtering
        self.data = self.data.reset_index(drop=True)

        # Prepare documents for LDA model (potentially combined)
        if combine_by_window:
            print(f"   Combining posts by user and time bin ({self.time_bin})...")
            # Create a temporary df for combining
            df_for_combine = self.data[['user_id', 'timestamp', 'tokens']].copy()

            # Add time bin column
            if self.time_bin == 'day':
                df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.date
            elif self.time_bin == 'week':
                df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('W').apply(lambda x: x.start_time)
            elif self.time_bin == 'month':
                df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('M').apply(lambda x: x.start_time)
            else:
                 # Default to week if time_bin is invalid
                 print(f"   Warning: Invalid time_bin '{self.time_bin}'. Defaulting to 'week'.")
                 df_for_combine['time_bin'] = df_for_combine['timestamp'].dt.to_period('W').apply(lambda x: x.start_time)

            # Combine posts by user and time bin
            grouped = df_for_combine.groupby(['user_id', 'time_bin'])

            combined_documents = []
            doc_meta_list = []

            for idx, ((user_id, time_bin), group) in enumerate(grouped):
                # Concatenate lists of tokens
                combined_tokens = [token for tokens_list in group['tokens'] for token in tokens_list]
                if combined_tokens:  # Only add if there are tokens after combining
                    combined_documents.append(combined_tokens)
                    # Use group mean timestamp, store original indices if needed?
                    doc_meta_list.append({
                        'user_id': user_id,
                        'time_bin': time_bin,
                        'timestamp': group['timestamp'].mean(), # Representative timestamp for the combined doc
                        'doc_id': idx # Simple index for combined documents
                    })

            self.documents = combined_documents # Documents for LDA
            self.doc_meta = pd.DataFrame(doc_meta_list) # Metadata for combined documents
            print(f"   Created {len(self.documents)} combined documents for LDA.")

        else:
            print("   Using each post as a separate document for LDA.")
            # Each post is a separate document for LDA
            # self.data already has the 'tokens' column correctly indexed
            self.documents = self.data['tokens'].tolist()
            # Metadata maps directly to self.data index
            self.doc_meta = self.data[['user_id', 'timestamp']].copy()
            self.doc_meta['doc_id'] = self.doc_meta.index # doc_id matches self.data index

        return self.documents # Return documents for LDA model

    def create_lda_model(self, documents: List[List[str]], passes: int = 20) -> Optional[LdaModel]:
        """
        Create and train an LDA topic model.

        Args:
            documents: List of preprocessed documents (lists of tokens)
            passes: Number of passes through the corpus during training

        Returns:
            Trained LDA model or None if failed
        """
        print("   Creating LDA dictionary and corpus...")
        if not documents:
             print("   Error: Cannot create LDA model with no documents.")
             return None
        try:
            # Create dictionary
            self.dictionary = corpora.Dictionary(documents)

            # Filter extremes (optional but recommended)
            print(f"   Initial dictionary size: {len(self.dictionary)}")
            self.dictionary.filter_extremes(no_below=3, no_above=0.6) # Keep tokens present in >=5 docs, <70% of docs
            print(f"   Dictionary size after filtering: {len(self.dictionary)}")
            if len(self.dictionary) == 0:
                 print("   Error: Dictionary became empty after filtering extremes. Check data or filter parameters.")
                 return None

            # Create document-term matrix (corpus)
            self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
            # Filter out empty documents from corpus and corresponding documents list if any were created
            valid_indices = [i for i, doc in enumerate(self.corpus) if doc]
            if len(valid_indices) < len(self.corpus):
                 print(f"   Warning: Removed {len(self.corpus) - len(valid_indices)} documents that became empty after dictionary filtering.")
                 self.corpus = [self.corpus[i] for i in valid_indices]
                 # Important: Adjust doc_meta if corpus changes size
                 if self.doc_meta is not None and len(self.doc_meta) == len(documents):
                     self.doc_meta = self.doc_meta.iloc[valid_indices].reset_index(drop=True)
                     # Update doc_id to match new corpus index
                     self.doc_meta['doc_id'] = self.doc_meta.index
                 else:
                     print("   Warning: Could not reliably adjust doc_meta after filtering empty documents.")
                 # Also adjust self.documents if needed later? For now, corpus is key.

            if not self.corpus:
                 print("   Error: Corpus is empty after filtering. Cannot train LDA model.")
                 return None

            print(f"   Training LDA model with {self.num_topics} topics on {len(self.corpus)} documents...")
            # Train LDA model
            self.lda_model = LdaModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                num_topics=self.num_topics,
                passes=passes,
                alpha='auto', # Let Gensim learn asymmetric alpha
                eta='auto',   # Let Gensim learn asymmetric eta
                random_state=42,
                eval_every=None, # Disable perplexity logging during training for speed
                iterations=100, # Increase iterations for potentially better convergence
                chunksize=2000 # Process documents in chunks
            )
            print("   LDA model training complete.")
            return self.lda_model

        except Exception as e:
            print(f"   Error during LDA model creation: {e}")
            self.lda_model = None
            self.dictionary = None
            self.corpus = None
            return None

    def get_document_topics(self) -> Optional[np.ndarray]:
        """
        Get topic distribution for each document in the corpus.

        Returns:
            numpy array of shape (n_documents, n_topics) or None if failed
        """
        if self.lda_model is None or self.corpus is None:
            print("   Error: LDA model or corpus not available for getting document topics.")
            return None

        print("   Calculating topic distribution for each document...")
        try:
            # Get topic distribution for each document
            num_docs = len(self.corpus)
            doc_topics = np.zeros((num_docs, self.num_topics), dtype=np.float32) # Use float32 for memory

            # Efficiently get distributions
            all_topics = self.lda_model.get_document_topics(self.corpus, minimum_probability=0.0)

            for i, doc_topic_list in enumerate(all_topics):
                for topic_idx, prob in doc_topic_list:
                    if 0 <= topic_idx < self.num_topics: # Bounds check
                        doc_topics[i, topic_idx] = prob
                    else:
                         print(f"   Warning: Invalid topic index {topic_idx} encountered for document {i}.")

            self.doc_topics = doc_topics
            print(f"   Calculated distributions for {num_docs} documents.")
            return doc_topics
        except Exception as e:
            print(f"   Error calculating document topics: {e}")
            self.doc_topics = None
            return None

    def calculate_user_topic_distributions(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Calculate aggregated topic distribution for each user.

        Returns:
            Dictionary mapping user_id to their topic distribution (mean), or None if failed.
        """
        if self.doc_topics is None:
            print("   Document topics not calculated yet. Running get_document_topics()...")
            if self.get_document_topics() is None:
                 print("   Error: Failed to get document topics. Cannot calculate user distributions.")
                 return None

        if self.doc_meta is None or 'user_id' not in self.doc_meta.columns or 'doc_id' not in self.doc_meta.columns:
             print("   Error: Document metadata (doc_meta) with user_id and doc_id is missing or invalid.")
             return None

        print("   Aggregating topic distributions per user...")
        user_topic_distributions = {}
        missing_doc_ids = 0

        # Ensure doc_topics has the expected shape
        if len(self.doc_topics) != self.doc_meta['doc_id'].max() + 1:
             print(f"   Warning: Mismatch between doc_topics length ({len(self.doc_topics)}) and max doc_id ({self.doc_meta['doc_id'].max()}). Some documents might be missing.")
             # Fallback: Use the minimum length? Or fail? Let's proceed with caution.

        for user_id in self.doc_meta['user_id'].unique():
            # Get doc_ids for this user from metadata
            user_doc_ids = self.doc_meta[self.doc_meta['user_id'] == user_id]['doc_id'].values

            # Filter doc_ids that are within the bounds of the doc_topics array
            valid_doc_ids = [doc_id for doc_id in user_doc_ids if 0 <= doc_id < len(self.doc_topics)]
            if len(valid_doc_ids) < len(user_doc_ids):
                 missing_doc_ids += (len(user_doc_ids) - len(valid_doc_ids))
                 # print(f"   Warning: User {user_id} has doc_ids outside doc_topics bounds.")

            if not valid_doc_ids:
                # print(f"   Warning: No valid document topic distributions found for user {user_id}.")
                continue # Skip user if no valid documents

            # Get topic distributions for these valid documents
            user_docs_topics = self.doc_topics[valid_doc_ids]

            # Aggregate topic distributions (mean)
            # Check for empty array before mean
            if user_docs_topics.size > 0 :
                 user_topic_dist = user_docs_topics.mean(axis=0)
                 user_topic_distributions[user_id] = user_topic_dist
            else:
                  print(f"   Warning: User {user_id} had valid doc IDs but resulted in empty topic array for mean calculation.")


        if missing_doc_ids > 0:
             print(f"   Total missing/out-of-bounds doc_ids during user aggregation: {missing_doc_ids}")

        self.user_topic_distributions = user_topic_distributions
        print(f"   Calculated distributions for {len(user_topic_distributions)} users.")
        return user_topic_distributions

    def calculate_gini_coefficient(self, array: np.ndarray) -> float:
        """Calculate the Gini coefficient for a distribution."""
        if array is None or np.sum(array) == 0: # Handle zero sum
            return np.nan # Or 0, depending on desired behavior for empty dists
        array = np.sort(np.abs(array)) # Use absolute value, handle potential negatives?
        n = len(array)
        if n <= 1: return 0.0 # Gini is 0 for single value
        index = np.arange(1, n + 1)
        # Gini coefficient formula
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))

    def calculate_shannon_entropy(self, array: np.ndarray) -> float:
        """Calculate Shannon entropy for a distribution."""
        if array is None: return np.nan
        # Ensure array is 1D
        array = np.asarray(array).flatten()
        # Filter out zero probabilities and normalize
        clean_array = array[array > 0]
        if clean_array.size == 0:
            return 0.0 # Entropy of empty distribution is 0
        normalized_array = clean_array / np.sum(clean_array)
        # Calculate entropy
        return entropy(normalized_array, base=2) # Use base 2 for bits

    def calculate_top_n_ratio(self, array: np.ndarray, n: int = 1) -> float:
        """Calculate the ratio of the top N topics in a distribution."""
        if array is None or n < 1: return np.nan
        total_sum = np.sum(array)
        if total_sum == 0: return 0.0 # Or NaN? Let's return 0 for empty dist
        sorted_array = np.sort(array)[::-1] # Sort descending
        top_n_sum = np.sum(sorted_array[:n])
        return top_n_sum / total_sum

    def calculate_user_narrowness_scores(self) -> Optional[pd.DataFrame]:
        """
        Calculate narrowness scores for each user using multiple metrics.

        Returns:
            DataFrame with user narrowness scores or None if failed.
        """
        if self.user_topic_distributions is None:
            print("   User topic distributions not calculated yet. Running calculate_user_topic_distributions()...")
            if self.calculate_user_topic_distributions() is None:
                 print("   Error: Failed to get user distributions. Cannot calculate narrowness.")
                 return None

        if self.doc_meta is None:
             print("   Error: Document metadata (doc_meta) required for post counts is missing.")
             return None


        print("   Calculating user narrowness metrics (Gini, Entropy, Top Ratios)...")
        narrowness_data = []
        post_counts = self.doc_meta['user_id'].value_counts().to_dict() # Efficiently get post counts

        for user_id, topic_dist in self.user_topic_distributions.items():
            post_count = post_counts.get(user_id, 0) # Get post count for this user

            # Calculate metrics
            gini = self.calculate_gini_coefficient(topic_dist)
            entropy_score = self.calculate_shannon_entropy(topic_dist)
            top1_ratio = self.calculate_top_n_ratio(topic_dist, n=1)
            top2_ratio = self.calculate_top_n_ratio(topic_dist, n=2)

            # Find dominant topic
            dominant_topic = np.argmax(topic_dist) if topic_dist.size > 0 else -1 # Handle empty dist

            narrowness_data.append({
                'user_id': user_id,
                'post_count': post_count,
                'gini_coefficient': gini,
                'shannon_entropy': entropy_score, # Lower entropy = more narrowness
                'top1_ratio': top1_ratio,
                'top2_ratio': top2_ratio,
                'dominant_topic': dominant_topic
            })

        if not narrowness_data:
             print("   Warning: No narrowness data generated.")
             self.user_narrowness_scores = pd.DataFrame(columns=[ # Create empty df with correct columns
                  'user_id', 'post_count', 'gini_coefficient', 'shannon_entropy',
                  'top1_ratio', 'top2_ratio', 'dominant_topic'])
             return self.user_narrowness_scores


        # Create DataFrame
        narrowness_df = pd.DataFrame(narrowness_data)

        # Sort by Gini coefficient (descending - higher Gini means more narrow)
        narrowness_df = narrowness_df.sort_values('gini_coefficient', ascending=False, na_position='last')

        self.user_narrowness_scores = narrowness_df
        print(f"   Calculated narrowness scores for {len(narrowness_df)} users.")
        return narrowness_df

    def calculate_temporal_topic_data(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        Calculate topic proportions over time for each user based on document metadata.

        Returns:
            Dictionary mapping user_id to DataFrame with temporal topic data, or None if failed.
        """
        if self.doc_topics is None:
            print("   Document topics not calculated yet. Running get_document_topics()...")
            if self.get_document_topics() is None:
                 print("   Error: Failed to get document topics. Cannot calculate temporal data.")
                 return None

        if self.doc_meta is None or not all(col in self.doc_meta.columns for col in ['user_id', 'timestamp', 'doc_id']):
            print("   Error: Document metadata (doc_meta) with user_id, timestamp, and doc_id is missing or invalid.")
            return None

        print("   Calculating temporal topic evolution for users...")
        temporal_data = {}
        processed_users = 0

        # --- Determine the correct time column based on how doc_meta was created ---
        if 'time_bin' in self.doc_meta.columns:
            # If we combined by window, doc_meta represents combined docs
            time_col_in_meta = 'time_bin'
            print(f"   Using pre-combined '{time_col_in_meta}' for temporal aggregation.")
            meta_df_for_temporal = self.doc_meta
        else:
            # If each post is separate, doc_meta represents original posts. We need to bin them.
            print(f"   Binning individual post timestamps using time_bin='{self.time_bin}'.")
            meta_df_for_temporal = self.doc_meta.copy() # Work on a copy
            meta_df_for_temporal['timestamp_dt'] = pd.to_datetime(meta_df_for_temporal['timestamp'], errors='coerce')
            meta_df_for_temporal = meta_df_for_temporal.dropna(subset=['timestamp_dt']) # Drop rows where timestamp couldn't be parsed

            if self.time_bin == 'day':
                meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.date
            elif self.time_bin == 'week':
                # Ensure it starts on a consistent day, e.g., Monday
                meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('W-MON').apply(lambda p: p.start_time.date())
            elif self.time_bin == 'month':
                meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('M').apply(lambda p: p.start_time.date())
            else:
                 print(f"   Warning: Invalid time_bin '{self.time_bin}'. Defaulting to 'week'.")
                 meta_df_for_temporal['time_period'] = meta_df_for_temporal['timestamp_dt'].dt.to_period('W-MON').apply(lambda p: p.start_time.date())
            time_col_in_meta = 'time_period' # The column containing the calculated period start
        # ------------------------------------------------------------------------

        # Group by user first, then process each user's data
        grouped_by_user = meta_df_for_temporal.groupby('user_id')

        for user_id, user_meta_df in grouped_by_user:
            user_temporal = {}
            # Group user's metadata by the calculated time period
            grouped_by_time = user_meta_df.groupby(time_col_in_meta)

            for time_period, group_df in grouped_by_time:
                # Get the original doc_ids for this user in this time period
                doc_ids_in_group = group_df['doc_id'].values

                # Filter doc_ids that are valid indices for doc_topics
                valid_doc_ids = [doc_id for doc_id in doc_ids_in_group if 0 <= doc_id < len(self.doc_topics)]

                if valid_doc_ids:
                    # Get the topic distributions for these valid documents
                    bin_topic_dist = self.doc_topics[valid_doc_ids]
                    # Calculate the mean distribution for this time bin
                    if bin_topic_dist.size > 0:
                         user_temporal[time_period] = bin_topic_dist.mean(axis=0)

            # Convert results for this user to a DataFrame
            if user_temporal:
                user_temporal_df = pd.DataFrame.from_dict(user_temporal, orient='index')
                user_temporal_df.columns = [f'Topic_{i}' for i in range(self.num_topics)]
                user_temporal_df.index.name = 'time_bin'
                # Convert index to datetime if it's not already (e.g., if it was date objects)
                user_temporal_df.index = pd.to_datetime(user_temporal_df.index)
                user_temporal_df = user_temporal_df.sort_index()
                temporal_data[user_id] = user_temporal_df
                processed_users += 1

        self.temporal_topic_data = temporal_data
        print(f"   Calculated temporal data for {processed_users} users.")
        return temporal_data


    def detect_suspicious_users(self, post_frequency_threshold: float = 0.95,
                               narrowness_threshold: float = 0.95,
                               similarity_threshold: float = 0.8, # Jaccard threshold
                               minhash_permutations: int = 128) -> Optional[pd.DataFrame]:
        """
        Flag suspicious users based on posting patterns and topic narrowness.
        Uses MinHashLSH for efficient approximate duplicate detection.

        Args:
            post_frequency_threshold: Percentile threshold for post frequency
            narrowness_threshold: Percentile threshold for topic narrowness (Gini coefficient)
            similarity_threshold: Jaccard similarity threshold for detecting duplicate posts
            minhash_permutations: Number of permutations for MinHash calculation.

        Returns:
            DataFrame with flagged suspicious users and their metrics or None if failed.
        """
        if not has_datasketch:
             print("   Warning: 'datasketch' library not installed. Skipping duplicate detection.")
             # Proceed without duplicate check if library is missing
             if self.user_narrowness_scores is None:
                if self.calculate_user_narrowness_scores() is None:
                    print("   Error: Failed to calculate narrowness scores. Cannot detect suspicious users.")
                    return None

             post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
             narrowness_cutoff = np.percentile(self.user_narrowness_scores['gini_coefficient'].dropna(), narrowness_threshold) # Drop NaN for percentile

             suspicious = self.user_narrowness_scores.copy()
             suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
             suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'] >= narrowness_cutoff
             suspicious['duplicate_post_ratio'] = 0.0 # Assign default
             suspicious['suspicious_duplicates'] = False # Assign default
             suspicious['suspicious'] = suspicious['suspicious_post_freq'] & suspicious['suspicious_narrowness']
             suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], ascending=[False, False, False])
             print("   Finished flagging based on frequency and narrowness only.")
             return suspicious


        if self.user_narrowness_scores is None:
            if self.calculate_user_narrowness_scores() is None:
                print("   Error: Failed to calculate narrowness scores. Cannot detect suspicious users.")
                return None

        # Check if 'tokens' column is available in self.data
        if self.data is None or 'tokens' not in self.data.columns:
             print("   Error: Precomputed 'tokens' column not found in self.data. Required for duplicate check.")
             print("   Run preprocess_data first or ensure it populates self.data['tokens'].")
             # Fallback: proceed without duplicate check
             print("   Proceeding with suspicious detection based on frequency and narrowness only.")
             post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
             narrowness_cutoff = np.percentile(self.user_narrowness_scores['gini_coefficient'].dropna(), narrowness_threshold)
             suspicious = self.user_narrowness_scores.copy()
             suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
             suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'] >= narrowness_cutoff
             suspicious['duplicate_post_ratio'] = 0.0
             suspicious['suspicious_duplicates'] = False
             suspicious['suspicious'] = suspicious['suspicious_post_freq'] & suspicious['suspicious_narrowness']
             suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], ascending=[False, False, False])
             return suspicious


        print("   Calculating suspicious flags (frequency, narrowness)...")
        post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
        # Handle potential NaNs in gini before percentile calculation
        valid_gini = self.user_narrowness_scores['gini_coefficient'].dropna()
        if valid_gini.empty:
            print("   Warning: No valid Gini coefficients found. Setting narrowness cutoff based on available data or default.")
            narrowness_cutoff = np.inf # Or some other default if all NaNs
        else:
            narrowness_cutoff = np.percentile(valid_gini, narrowness_threshold)

        suspicious = self.user_narrowness_scores.copy()
        suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
        # Apply cutoff carefully if NaNs exist
        suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'].apply(lambda x: x >= narrowness_cutoff if pd.notna(x) else False)


        # --- Efficient Duplicate Detection using MinHashLSH ---
        print("   Calculating duplicate post ratios using MinHashLSH...")
        duplicate_post_ratios = {}
        processed_users_for_duplicates = 0

        # Iterate through users who have posts (based on narrowness scores)
        users_to_check = suspicious['user_id'].unique()

        for user_id in users_to_check:
            # Retrieve pre-computed token lists for the user directly from self.data
            # Ensure index alignment if self.data was modified after token creation
            user_rows = self.data[self.data['user_id'] == user_id]
            if user_rows.empty:
                 duplicate_post_ratios[user_id] = 0.0
                 continue

            user_token_lists = user_rows['tokens'].tolist()

            # Filter out None or potentially non-list entries
            user_token_lists = [tokens for tokens in user_token_lists if isinstance(tokens, list)]

            if len(user_token_lists) <= 1:
                duplicate_post_ratios[user_id] = 0.0
                continue

            # Create MinHash objects for each post
            minhashes = {}
            token_sets = {} # Store sets for exact Jaccard check later
            post_keys = [] # Keep track of valid post keys (using DataFrame index is safer)

            for i, tokens in enumerate(user_token_lists):
                 # Use the original index from user_rows as a stable key
                 original_index = user_rows.index[i]
                 key = f"post_{original_index}" # Unique key based on original index

                 if not tokens: # Skip empty posts
                     continue

                 post_keys.append(key)
                 token_set = set(tokens)
                 # Handle potential empty sets after tokenization/filtering
                 if not token_set:
                      continue

                 token_sets[key] = token_set # Store the set

                 m = MinHash(num_perm=minhash_permutations)
                 for word in token_set:
                     m.update(word.encode('utf8')) # Encode words for MinHash
                 minhashes[key] = m


            if len(minhashes) <= 1: # Check if any valid posts remained
                duplicate_post_ratios[user_id] = 0.0
                continue

            # Create LSH index with the desired Jaccard threshold
            # Adjust threshold slightly if needed based on MinHash properties
            lsh = MinHashLSH(threshold=similarity_threshold, num_perm=minhash_permutations)

            # Index MinHash objects (handle potential key errors if a post had no tokens)
            indexed_keys = set()
            for key, m in minhashes.items():
                 lsh.insert(key, m)
                 indexed_keys.add(key)


            # Find candidate pairs and verify with exact Jaccard
            similar_pairs_count = 0
            checked_pairs = set() # Avoid double counting (i,j) and (j,i)

            # Query only for keys that were successfully indexed
            for key in indexed_keys:
                 # Ensure the key exists in minhashes before querying (safety check)
                 if key not in minhashes: continue
                 query_minhash = minhashes[key]

                 try:
                      result = lsh.query(query_minhash)
                 except KeyError:
                      print(f"   Warning: Key {key} not found in LSH index during query (User: {user_id}). Skipping.")
                      continue


                 # result contains keys of candidate similar posts (including the query key itself)
                 for other_key in result:
                     # Ensure other_key was also indexed and exists in token_sets
                     if other_key not in indexed_keys or other_key not in token_sets:
                          continue

                     # Avoid self-comparison and comparing pairs twice
                     pair = tuple(sorted((key, other_key)))
                     if key == other_key or pair in checked_pairs:
                         continue
                     checked_pairs.add(pair)

                     # --- Optional but Recommended: Verify with exact Jaccard ---
                     set1 = token_sets.get(key) # Use .get() for safety
                     set2 = token_sets.get(other_key)

                     # Ensure both sets were retrieved successfully
                     if set1 is None or set2 is None:
                          continue

                     union_len = len(set1.union(set2))
                     if union_len == 0:
                         actual_jaccard = 0.0
                     else:
                         intersection_len = len(set1.intersection(set2))
                         actual_jaccard = intersection_len / union_len

                     if actual_jaccard >= similarity_threshold:
                         similar_pairs_count += 1
                     # ----------------------------------------------------------


            # Calculate the ratio
            num_valid_posts = len(minhashes)
            total_possible_pairs = num_valid_posts * (num_valid_posts - 1) / 2 if num_valid_posts > 1 else 0
            duplicate_ratio = similar_pairs_count / max(1, total_possible_pairs)
            duplicate_post_ratios[user_id] = duplicate_ratio
            processed_users_for_duplicates += 1

        print(f"   Calculated duplicate ratios for {processed_users_for_duplicates} users.")

        # Add the calculated ratios to the DataFrame
        suspicious['duplicate_post_ratio'] = suspicious['user_id'].map(duplicate_post_ratios).fillna(0.0) # Fill NaNs just in case
        suspicious['suspicious_duplicates'] = suspicious['duplicate_post_ratio'] > 0.5 # Use a threshold (e.g., 50% of pairs are similar)

        # --- End MinHashLSH Section ---


        # Overall suspicion flag
        suspicious['suspicious'] = (suspicious['suspicious_post_freq'] &
                                   suspicious['suspicious_narrowness']) | suspicious['suspicious_duplicates']

        # Sort by suspiciousness
        suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'],
                                           ascending=[False, False, False], na_position='last') # Handle potential NaNs in gini

        print(f"   Flagged {suspicious['suspicious'].sum()} users as suspicious.")
        return suspicious


    # --- OLD O(N^2) Jaccard Calculation - Commented Out ---
    # def detect_suspicious_users_OLD(self, post_frequency_threshold: float = 0.95,
    #                            narrowness_threshold: float = 0.95,
    #                            similarity_threshold: float = 0.8) -> pd.DataFrame:
    #     """
    #     Flag suspicious users based on posting patterns and topic narrowness.
    #     (Original O(N^2) Jaccard version)
    #     """
    #     if self.user_narrowness_scores is None:
    #         self.calculate_user_narrowness_scores()

    #     post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
    #     narrowness_cutoff = np.percentile(self.user_narrowness_scores['gini_coefficient'], narrowness_threshold)

    #     suspicious = self.user_narrowness_scores.copy()
    #     suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
    #     suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'] >= narrowness_cutoff

    #     # Calculate similarity between posts (this can be computationally expensive)
    #     duplicate_post_ratios = {}

    #     for user_id in suspicious['user_id']: # Loop 1: Iterates through ALL users initially
    #         # Assuming self.data holds the original posts accessible via user_id
    #         user_posts = self.data[self.data['user_id'] == user_id]['post_content'].tolist()

    #         if len(user_posts) <= 1:
    #             duplicate_post_ratios[user_id] = 0
    #             continue

    #         similar_pairs = 0
    #         total_pairs = 0

    #         # Loop 2: Outer loop for pairs
    #         for i in range(len(user_posts)):
    #             # Loop 3: Inner loop for pairs - THIS IS THE PROBLEM AREA
    #             for j in range(i+1, len(user_posts)):
    #                 total_pairs += 1

    #                 # Tokenize posts (happens INSIDE the innermost loop)
    #                 tokens1 = set(self.preprocess_text(user_posts[i])) # Costly preprocessing
    #                 tokens2 = set(self.preprocess_text(user_posts[j])) # Costly preprocessing

    #                 if not tokens1 or not tokens2:
    #                     continue

    #                 # Calculate Jaccard similarity
    #                 intersection_len = len(tokens1.intersection(tokens2))
    #                 union_len = len(tokens1.union(tokens2))

    #                 if union_len == 0:
    #                      jaccard = 0.0
    #                 else:
    #                      jaccard = intersection_len / union_len


    #                 if jaccard > similarity_threshold:
    #                     similar_pairs += 1

    #         duplicate_ratio = similar_pairs / max(1, total_pairs)
    #         duplicate_post_ratios[user_id] = duplicate_ratio

    #     suspicious['duplicate_post_ratio'] = suspicious['user_id'].map(duplicate_post_ratios)
    #     suspicious['suspicious_duplicates'] = suspicious['duplicate_post_ratio'] > 0.5 # Or use a different threshold

    #     # Overall suspicion flag
    #     suspicious['suspicious'] = (suspicious['suspicious_post_freq'] &
    #                                suspicious['suspicious_narrowness']) | suspicious['suspicious_duplicates']

    #     # Sort by suspiciousness
    #     suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'],
    #                                        ascending=[False, False, False])

    #     return suspicious
    # --- End OLD Version ---


    def visualize_topic_words(self, n_words: int = 10, save_path: Optional[str] = None):
        """
        Visualize top words for each topic using word clouds.

        Args:
            n_words: Number of words to show per topic
            save_path: Path to save the visualization
        """
        if self.lda_model is None:
            print("   Error: LDA model not created. Cannot visualize topic words.")
            return

        if not has_wordcloud:
            print("   WordCloud package not installed. Skipping visualization.")
            return

        print("   Generating topic word clouds...")
        # Set up the figure
        n_cols = min(5, self.num_topics)
        n_rows = (self.num_topics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(min(20, 5 * n_cols), 4 * n_rows))
        # Handle case where num_topics < n_cols or only 1 topic
        if n_rows == 1 and n_cols == 1:
             axes = [axes]
        else:
             axes = axes.flatten()

        # Create word clouds for each topic
        shown_topics = 0
        for i in range(self.num_topics):
            if i >= len(axes):
                break

            try:
                # Get top words and their weights
                top_words_list = self.lda_model.show_topic(i, n_words)
                top_words = dict(top_words_list)

                if not top_words:
                     print(f"   Warning: No words found for Topic {i}. Skipping word cloud.")
                     axes[i].set_title(f'Topic {i}\n(No words)')
                     axes[i].axis('off')
                     continue

                # Create word cloud
                wc = WordCloud(width=400, height=300, background_color='white',
                               max_words=n_words, colormap='viridis', prefer_horizontal=0.9)
                wc.generate_from_frequencies(top_words)

                # Plot
                axes[i].imshow(wc, interpolation='bilinear')
                axes[i].set_title(f'Topic {i}')
                axes[i].axis('off')
                shown_topics += 1
            except Exception as e:
                 print(f"   Error generating word cloud for Topic {i}: {e}")
                 axes[i].set_title(f'Topic {i}\n(Error)')
                 axes[i].axis('off')


        # Hide unused subplots
        for j in range(shown_topics, len(axes)):
            axes[j].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97]) # Adjust layout

        if save_path:
            try:
                plt.savefig(save_path, dpi=150)
                print(f"   Topic word clouds saved to {save_path}")
            except Exception as e:
                print(f"   Error saving word clouds figure: {e}")

        plt.show() # Display the plot
        plt.close(fig) # Close the figure to free memory


    def visualize_user_topic_pie(self, user_id: str, save_path: Optional[str] = None):
        """
        Visualize topic distribution for a user as a pie chart.

        Args:
            user_id: ID of the user to visualize
            save_path: Path to save the visualization
        """
        if self.user_topic_distributions is None:
            print(f"   User topic distributions not calculated. Cannot visualize pie for {user_id}.")
            return

        if user_id not in self.user_topic_distributions:
            print(f"   User {user_id} not found in calculated distributions.")
            return

        # Get topic distribution for this user
        topic_dist = self.user_topic_distributions[user_id]

        if topic_dist is None or np.all(topic_dist == 0):
             print(f"   User {user_id} has no significant topic distribution to visualize.")
             return


        # Only include topics with significant contribution
        threshold = 0.03
        significant_indices = np.where(topic_dist > threshold)[0]
        significant_probs = topic_dist[significant_indices]

        if len(significant_indices) == 0:
            # If no topic is above threshold, show the top one or two
             significant_indices = np.argsort(topic_dist)[-2:] # Top 2
             significant_probs = topic_dist[significant_indices]
             # Filter out zeros if top ones are zero
             non_zero_mask = significant_probs > 1e-6
             significant_indices = significant_indices[non_zero_mask]
             significant_probs = significant_probs[non_zero_mask]
             if len(significant_indices) == 0:
                  print(f"   User {user_id} distribution is effectively zero. Cannot create pie chart.")
                  return


        other_prob = 1.0 - np.sum(significant_probs) # Calculate 'Other' based on displayed slices

        # Sort by probability for consistent ordering
        sort_order = np.argsort(significant_probs)[::-1]
        significant_indices = significant_indices[sort_order]
        significant_probs = significant_probs[sort_order]


        # Create labels and values
        labels = [f'Topic {i}' for i in significant_indices]
        values = significant_probs.tolist()

        if other_prob > 0.01: # Only add 'Other' slice if it's reasonably large
            labels.append('Other')
            values.append(other_prob)

        # Create pie chart
        fig, ax = plt.subplots(figsize=(8, 8)) # Adjusted size
        # Use a perceptually uniform colormap if many slices, or tab20 for fewer
        colors = plt.cm.viridis(np.linspace(0, 1, len(values))) if len(values) > 20 else plt.cm.tab20.colors[:len(values)]

        wedges, texts, autotexts = ax.pie(values, autopct='%1.1f%%', startangle=90, colors=colors,
                                           pctdistance=0.85) # Adjust label position

        # Improve autopct text visibility
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(9)
            autotext.set_fontweight('bold')


        ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(f'Topic Distribution for User {user_id}', fontsize=14)

        # Add legend
        ax.legend(wedges, labels,
                  title="Topics",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1), fontsize=9)

        plt.tight_layout()


        if save_path:
            try:
                plt.savefig(save_path, bbox_inches='tight')
                # print(f"   User topic pie chart saved to {save_path}") # Less verbose
            except Exception as e:
                print(f"   Error saving user pie chart {save_path}: {e}")

        plt.show() # Display the plot
        plt.close(fig) # Close the figure

    def visualize_user_temporal_topics(self, user_id: str, save_path: Optional[str] = None):
        """
        Visualize temporal topic distribution for a user as a stacked area chart.

        Args:
            user_id: ID of the user to visualize
            save_path: Path to save the visualization
        """
        if self.temporal_topic_data is None:
            print(f"   Temporal topic data not calculated. Cannot visualize for {user_id}.")
            return

        if user_id not in self.temporal_topic_data:
            print(f"   User {user_id} not found in temporal data.")
            return

        # Get temporal data for this user
        user_temporal_df = self.temporal_topic_data[user_id]

        if user_temporal_df.empty:
             print(f"   User {user_id} has no temporal data to visualize.")
             return

        # Only include top N topics consistently over time + 'Other'
        top_n = min(5, self.num_topics) # Show top 5 topics overall for this user
        avg_topic_dist = user_temporal_df.mean().nlargest(top_n)
        top_topic_cols = avg_topic_dist.index.tolist()

        # Ensure all top columns exist
        plot_df = user_temporal_df.copy()
        for col in top_topic_cols:
             if col not in plot_df:
                  print(f"   Warning: Expected top topic column {col} not found in temporal data for user {user_id}.")
                  top_topic_cols.remove(col) # Remove if missing


        # Create 'Other' category for remaining topics
        other_cols = [col for col in user_temporal_df.columns if col not in top_topic_cols]
        if other_cols:
            plot_df['Other'] = user_temporal_df[other_cols].sum(axis=1)
            plot_cols = top_topic_cols + ['Other']
        else:
            plot_cols = top_topic_cols # No 'Other' needed if top_n >= num_topics


        # Plot stacked area chart
        fig, ax = plt.subplots(figsize=(12, 6))
        try:
             plot_df[plot_cols].plot.area(stacked=True, alpha=0.8, ax=ax,
                                          colormap='tab10' if len(plot_cols) <= 10 else 'viridis') # Adjust colormap

             ax.set_title(f'Topic Evolution Over Time for User {user_id}', fontsize=14)
             ax.set_xlabel('Time', fontsize=12)
             ax.set_ylabel('Topic Proportion', fontsize=12)
             ax.legend(title='Topics', loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
             ax.grid(True, alpha=0.3)
             ax.set_ylim(0, 1) # Ensure y-axis is 0 to 1

             # Improve date formatting on x-axis if index is datetime
             if pd.api.types.is_datetime64_any_dtype(plot_df.index):
                 fig.autofmt_xdate() # Auto format dates


             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend


             if save_path:
                 try:
                     plt.savefig(save_path, bbox_inches='tight')
                     # print(f"   User temporal plot saved to {save_path}") # Less verbose
                 except Exception as e:
                     print(f"   Error saving user temporal plot {save_path}: {e}")

             plt.show() # Display the plot

        except Exception as plot_err:
            print(f"   Error generating temporal plot for user {user_id}: {plot_err}")
        finally:
            plt.close(fig) # Ensure figure is closed

    def visualize_narrowness_vs_frequency(self, metric: str = 'gini_coefficient',
                                         save_path: Optional[str] = None):
        """
        Visualize user narrowness vs. post frequency.

        Args:
            metric: Narrowness metric ('gini_coefficient', 'shannon_entropy', 'top1_ratio')
            save_path: Path to save the visualization
        """
        if self.user_narrowness_scores is None or self.user_narrowness_scores.empty:
            print("   User narrowness scores not calculated or empty. Cannot visualize.")
            return

        valid_metrics = ['gini_coefficient', 'shannon_entropy', 'top1_ratio', 'top2_ratio']
        if metric not in valid_metrics:
             print(f"   Invalid metric '{metric}'. Choose from: {valid_metrics}")
             metric = 'gini_coefficient' # Default to gini
             print(f"   Defaulting to '{metric}'.")

        if metric not in self.user_narrowness_scores.columns:
             print(f"   Metric column '{metric}' not found in narrowness scores.")
             return


        plot_data = self.user_narrowness_scores[['user_id', 'post_count', metric]].copy()
        # Handle potential NaNs in the chosen metric
        plot_data = plot_data.dropna(subset=[metric, 'post_count'])

        if plot_data.empty:
             print(f"   No valid data points found for metric '{metric}' after dropping NaNs.")
             return

        # Add suspicious flag if available
        is_suspicious_col = 'suspicious' in self.user_narrowness_scores.columns
        if is_suspicious_col:
             plot_data = plot_data.merge(self.user_narrowness_scores[['user_id', 'suspicious']], on='user_id', how='left')
             plot_data['suspicious'] = plot_data['suspicious'].fillna(False) # Assume not suspicious if merge fails
        else:
             plot_data['suspicious'] = False # Default if column doesn't exist

        print(f"   Visualizing {metric} vs Post Frequency...")
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by suspiciousness if available
        colors = ['red' if s else 'blue' for s in plot_data['suspicious']]
        sizes = [50 if s else 20 for s in plot_data['suspicious']] # Make suspicious points larger
        alpha = 0.6

        ax.scatter(plot_data['post_count'],
                   plot_data[metric],
                   c=colors, alpha=alpha, s=sizes, edgecolors='w', linewidths=0.5)

        # Dynamic Title
        title = f'User Topic Narrowness ({metric.replace("_", " ").title()}) vs. Post Frequency'
        if is_suspicious_col:
            suspicious_count = plot_data['suspicious'].sum()
            title += f'\n({suspicious_count} Suspicious Users Highlighted)'
        ax.set_title(title, fontsize=14)

        ax.set_xlabel('Number of Posts (Log Scale)', fontsize=12)
        ax.set_ylabel(f'Topic Narrowness ({metric.replace("_", " ").title()})', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Use log scale for post count, handle potential zero/negative values
        min_post_count = plot_data['post_count'].min()
        if min_post_count <= 0:
            print("   Warning: Post counts include zero or negative values, log scale might be problematic.")
            # Optional: Add small epsilon or use symlog? For now, standard log.
            ax.set_xscale('log')
        else:
            ax.set_xscale('log')

        # Add legend manually for colors if suspicious flag is used
        if is_suspicious_col:
             from matplotlib.lines import Line2D
             legend_elements = [Line2D([0], [0], marker='o', color='w', label='Not Suspicious',
                                       markerfacecolor='blue', markersize=8, alpha=alpha),
                                Line2D([0], [0], marker='o', color='w', label='Suspicious',
                                       markerfacecolor='red', markersize=10, alpha=alpha)]
             ax.legend(handles=legend_elements, loc='best', fontsize=9)

        plt.tight_layout()

        if save_path:
            try:
                plt.savefig(save_path, dpi=150)
                print(f"   Narrowness vs Frequency plot saved to {save_path}")
            except Exception as e:
                print(f"   Error saving narrowness plot: {e}")

        plt.show()
        plt.close(fig)


    def visualize_topic_embedding(self, method: str = 'tsne', save_path: Optional[str] = None):
        """
        Visualize document topic embeddings using dimensionality reduction.

        Args:
            method: Dimensionality reduction method ('tsne' or 'umap')
            save_path: Path to save the visualization
        """
        if self.doc_topics is None or self.doc_topics.size == 0:
            print("   Document topics not calculated or empty. Cannot visualize embeddings.")
            return

        if not has_dim_reduction:
            print("   UMAP or scikit-learn not installed. Skipping embedding visualization.")
            return

        print(f"   Performing {method.upper()} dimensionality reduction for topic embedding visualization...")
        # Apply dimensionality reduction
        try:
            if method.lower() == 'tsne':
                # Consider perplexity setting based on data size
                perplexity = min(30.0, max(5.0, (len(self.doc_topics)-1)/3.0)) # Heuristic
                print(f"     Using t-SNE with perplexity={perplexity:.1f}")
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=300, n_jobs=-1)
            elif method.lower() == 'umap':
                print("     Using UMAP")
                reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1, metric='hellinger') # Hellinger for distributions
            else:
                print(f"   Invalid method '{method}'. Choose 'tsne' or 'umap'. Defaulting to 'tsne'.")
                reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300, n_jobs=-1)

            # Get document embeddings
            doc_topics_embedded = reducer.fit_transform(self.doc_topics)

        except Exception as e:
            print(f"   Error during {method.upper()} dimensionality reduction: {e}")
            return


        # Get dominant topic for each document for coloring
        dominant_topics = np.argmax(self.doc_topics, axis=1)

        print("   Generating embedding scatter plot...")
        fig, ax = plt.subplots(figsize=(12, 10))

        # Color by topic using a categorical colormap
        num_unique_topics = len(np.unique(dominant_topics))
        cmap = plt.cm.get_cmap('viridis', num_unique_topics) if num_unique_topics > 20 else plt.cm.get_cmap('tab20', num_unique_topics)

        scatter = ax.scatter(doc_topics_embedded[:, 0], doc_topics_embedded[:, 1],
                             c=dominant_topics, cmap=cmap, alpha=0.6, s=20, # Smaller points
                             edgecolors='none') # No edges for dense plots

        ax.set_title(f'Document Topic Embedding ({method.upper()})', fontsize=14)
        ax.set_xlabel('Dimension 1', fontsize=12)
        ax.set_ylabel('Dimension 2', fontsize=12)
        ax.grid(True, alpha=0.2)

        # Add legend for topics
        # Create proxy artists for legend if too many topics
        if num_unique_topics <= 20:
             legend1 = ax.legend(*scatter.legend_elements(num=num_unique_topics),
                                  loc="center left", bbox_to_anchor=(1, 0.5), title="Topics", fontsize=9)
             ax.add_artist(legend1)
             plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
        else:
             print("   (Legend omitted due to large number of topics)")
             plt.tight_layout()


        if save_path:
            try:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"   Topic embedding plot saved to {save_path}")
            except Exception as e:
                print(f"   Error saving embedding plot: {e}")

        plt.show()
        plt.close(fig)


    def export_user_topic_data(self, output_path: str):
        """
        Export user topic distribution and narrowness scores to CSV.

        Args:
            output_path: Path to save the CSV file
        """
        if self.user_narrowness_scores is None or self.user_narrowness_scores.empty:
            print("   User narrowness scores not calculated or empty. Cannot export user topic data.")
            # Optionally create an empty file?
            Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            pd.DataFrame().to_csv(output_path, index=False)
            return

        print(f"   Exporting user topic data to {output_path}...")
        # Create a copy of the narrowness DataFrame
        export_df = self.user_narrowness_scores.copy()

        # Add full topic distribution for each user
        # Check if distributions exist before trying to access
        if self.user_topic_distributions:
            # Determine number of topics from the first user's distribution
            first_user = next(iter(self.user_topic_distributions))
            n_topics_dist = len(self.user_topic_distributions[first_user]) if first_user else self.num_topics

            topic_cols = {}
            for i in range(n_topics_dist):
                 topic_col_name = f'Topic_{i}'
                 # Efficiently map distributions using .get for missing users
                 topic_cols[topic_col_name] = export_df['user_id'].map(
                      lambda uid: self.user_topic_distributions.get(uid, np.zeros(n_topics_dist))[i] # Default to zero array
                 )
            export_df = pd.concat([export_df, pd.DataFrame(topic_cols)], axis=1)
        else:
             print("   Warning: User topic distributions not available to add to export.")


        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # Export to CSV
        try:
            export_df.to_csv(output_path, index=False, float_format='%.6f') # Format floats
            print(f"   Successfully exported user topic data.")
        except Exception as e:
            print(f"   Error exporting user topic data to {output_path}: {e}")


    def export_topic_words(self, output_path: str, n_words: int = 20):
        """
        Export top words for each topic to CSV.

        Args:
            output_path: Path to save the CSV file
            n_words: Number of top words to export for each topic
        """
        if self.lda_model is None:
            print("   Warning: LDA model not created. Cannot export topic words.")
            # Create an empty file or handle appropriately
            Path(output_path).parent.mkdir(parents=True, exist_ok=True) # Ensure dir exists
            pd.DataFrame(columns=['topic_id', 'word', 'weight', 'rank']).to_csv(output_path, index=False)
            return

        print(f"   Exporting top {n_words} words per topic to {output_path}...")
        topic_words = []
        for topic_id in range(self.num_topics):
            try:
                # Get words and weights using LdaModel method
                words_and_weights = self.lda_model.get_topic_terms(topic_id, topn=n_words)
                # Convert word IDs back to words
                words = [(self.dictionary[word_id], weight) for word_id, weight in words_and_weights]

                for rank, (word, weight) in enumerate(words):
                    topic_words.append({
                        'topic_id': topic_id,
                        'word': word,
                        'weight': weight,
                        'rank': rank + 1
                    })
            except IndexError:
                 print(f"   Warning: Word ID not found in dictionary for topic {topic_id}. Skipping some words.")
            except Exception as e:
                 print(f"   Warning: Could not get words for topic {topic_id}: {e}")

        # Create DataFrame and save to CSV
        topic_words_df = pd.DataFrame(topic_words)
        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            topic_words_df.to_csv(output_path, index=False, float_format='%.6f')
            print(f"   Successfully exported topic words.")
        except Exception as e:
            print(f"   Error exporting topic words to {output_path}: {e}")


    def run_full_pipeline(self,
                         input_file: str,
                         output_dir: str,
                         combine_by_window: bool = True,
                         visualize_topics: bool = True,
                         detect_suspicious: bool = True):
        """
        Run the complete analysis pipeline.
        Includes coherence score calculation and saving.
        """
        start_time = datetime.now()
        print(f"\n--- Starting LDA Pipeline: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Input file: {input_file}")
        print(f"Output dir: {output_dir}")
        print(f"Combine by window: {combine_by_window}, Time bin: {self.time_bin}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True) # Ensure output dir exists

        results = {} # To store key results

        # --- Pipeline Steps ---
        try:
            print("\n1. Loading data...")
            self.load_data(input_file)
            if self.data is None or self.data.empty: raise ValueError("Data loading failed or resulted in empty DataFrame.")
            results['initial_records'] = len(self.data)

            print("\n2. Preprocessing data...")
            # Preprocess_data now updates self.data and self.documents, self.doc_meta
            self.preprocess_data(combine_by_window=combine_by_window)
            if self.documents is None or not self.documents: raise ValueError("Preprocessing failed or resulted in no documents.")
            results['records_after_preprocessing'] = len(self.data) # Records in self.data
            results['documents_for_lda'] = len(self.documents) # Docs fed to LDA


            print(f"\n3. Creating LDA model with {self.num_topics} topics...")
            self.create_lda_model(self.documents)
            if self.lda_model is None: raise ValueError("LDA model creation failed.")

            print("\n4. Calculating document topics...")
            if self.get_document_topics() is None: raise ValueError("Document topic calculation failed.")

            print("\n5. Calculating user topic distributions...")
            if self.calculate_user_topic_distributions() is None: print("Warning: Could not calculate user distributions.")
            results['num_users_processed'] = len(self.user_topic_distributions) if self.user_topic_distributions else 0

            print("\n6. Calculating user narrowness scores...")
            if self.calculate_user_narrowness_scores() is None: print("Warning: Could not calculate narrowness scores.")

            print("\n7. Calculating temporal topic data...")
            if self.calculate_temporal_topic_data() is None: print("Warning: Could not calculate temporal data.")

            # --- Calculate and Save Coherence Score ---
            print("\n8. Calculating Topic Coherence...")
            coherence_score = np.nan # Default to NaN
            try:
                if self.lda_model and self.dictionary and self.documents: # Use self.documents
                    # Ensure documents list is not empty and filter if needed
                    valid_docs = [doc for doc in self.documents if doc]
                    if valid_docs and self.corpus: # Also check if corpus exists
                        print(f"   Calculating c_v coherence on {len(valid_docs)} documents...")
                        coherence_model_lda = CoherenceModel(model=self.lda_model,
                                                             texts=valid_docs, # Use the tokenized documents
                                                             dictionary=self.dictionary,
                                                             coherence='c_v')
                        coherence_score = coherence_model_lda.get_coherence()
                        print(f"   LDA Coherence Score (c_v): {coherence_score:.4f}")
                        results['coherence_score_c_v'] = coherence_score
                    else:
                        print("   Warning: No valid documents/corpus found for coherence calculation.")
                else:
                    print("   Warning: LDA model, dictionary, or documents not available for coherence calculation.")

                # Save coherence score
                coherence_data = {'coherence_mean': coherence_score if pd.notna(coherence_score) else None} # Save None if NaN for JSON
                coherence_path = output_path / "topic_coherence.json"
                with open(coherence_path, 'w') as f:
                    # Use a handler for potential numpy types
                    json.dump(coherence_data, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float_, np.float16, np.float32, np.float64)) else (None if pd.isna(x) else x))
                print(f"   Coherence score saved to {coherence_path}")

            except Exception as e:
                print(f"   Error calculating or saving coherence: {e}")
                results['coherence_score_c_v'] = None
                # Still try to save NaN/None if calculation failed
                try:
                     coherence_data = {'coherence_mean': None}
                     coherence_path = output_path / "topic_coherence.json"
                     with open(coherence_path, 'w') as f: json.dump(coherence_data, f, indent=2)
                except Exception as e_save:
                     print(f"   Failed to save None coherence score either: {e_save}")
            # --- End Coherence Calculation ---

            step_num = 9 # Adjust step numbers
            if detect_suspicious:
                print(f"\n{step_num}. Detecting suspicious users...")
                suspicious_users = self.detect_suspicious_users() # Uses MinHashLSH now
                if suspicious_users is not None:
                     suspicious_users.to_csv(output_path / "suspicious_users.csv", index=False)
                     results['suspicious_users_detected'] = suspicious_users['suspicious'].sum()
                     print(f"   Identified {results['suspicious_users_detected']} suspicious users")
                else:
                     print("   Warning: Suspicious user detection failed.")
                step_num += 1
            else:
                 print(f"\n{step_num}. Skipping suspicious user detection.")
                 step_num += 1


            print(f"\n{step_num}. Exporting results...")
            step_num += 1
            self.export_user_topic_data(output_path / "user_topic_data.csv")
            self.export_topic_words(output_path / "topic_words.csv")


            if visualize_topics:
                print(f"\n{step_num}. Creating visualizations...")
                step_num += 1
                # Topic word clouds
                self.visualize_topic_words(save_path=output_path / "topic_words.png")
                # User narrowness vs frequency
                self.visualize_narrowness_vs_frequency(save_path=output_path / "narrowness_vs_frequency.png")
                # Topic embedding
                self.visualize_topic_embedding(save_path=output_path / "topic_embedding.png")

                # Create visualizations for top users (if narrowness scores exist)
                print("   Generating visualizations for top 5 users (by Gini)...")
                if self.user_narrowness_scores is not None and not self.user_narrowness_scores.empty:
                     # Ensure user_ids are strings if needed by visualization functions
                     top_users = self.user_narrowness_scores.head(5)['user_id'].astype(str).tolist()
                     for user_id in top_users:
                         self.visualize_user_topic_pie(user_id, save_path=output_path / f"user_{user_id}_topics.png")
                         self.visualize_user_temporal_topics(user_id, save_path=output_path / f"user_{user_id}_temporal.png")
                else:
                     print("   Skipping top user visualizations as narrowness scores are unavailable.")
            else:
                 print(f"\n{step_num}. Skipping visualizations.")


        except Exception as pipeline_error:
             print(f"\n--- ERROR: LDA Pipeline failed ---")
             print(f"Error details: {pipeline_error}")
             import traceback
             traceback.print_exc() # Print full traceback
             print("------------------------------------")
             return None # Indicate failure


        end_time = datetime.now()
        print(f"\n--- LDA Pipeline finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')} ---")
        print(f"Total execution time: {end_time - start_time}")
        print("------------------------------------------------------")

        # Final summary results dictionary
        final_results = {
            'num_documents': results.get('documents_for_lda', 0),
            'num_users': results.get('num_users_processed', 0),
            'num_topics': self.num_topics,
            'output_dir': str(output_dir), # Ensure it's a string
            'coherence_score': results.get('coherence_score_c_v', None)
        }
        if detect_suspicious:
             final_results['suspicious_users_detected'] = results.get('suspicious_users_detected', 0)

        return final_results


# Example usage block (optional)
# if __name__ == "__main__":
#     # Initialize the system
#     topic_system = UserTopicModelingSystem(
#         num_topics=15,
#         time_bin='week',
#         lemmatize=True,
#         extra_stopwords=['rt', 'http', 'https', 'amp', 'via'] # Add relevant stopwords
#     )
#
#     # Create dummy data if needed for testing
#     # ... (create dummy preprocessed_data/dummy_data.csv)
#
#     # Run the pipeline
#     topic_system.run_full_pipeline(
#         input_file="preprocessed_data/reddit_processed.csv", # Replace with your data file
#         output_dir="topic_analysis_results/lda_results_test",
#         combine_by_window=False, # Test without combining
#         visualize_topics=True,
#         detect_suspicious=True
#     )