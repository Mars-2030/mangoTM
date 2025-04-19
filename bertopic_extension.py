import pandas as pd
import numpy as np
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

# Topic modeling with BERTopic
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer

# For measuring inequality/concentration
from scipy.stats import entropy
import matplotlib.cm as cm
from wordcloud import WordCloud

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


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
                 min_cluster_size: int = 15):
        """
        Initialize the BERTopic modeling system.
        
        Args:
            language_model: Sentence transformer model to use
            time_bin: Time binning for temporal analysis ('day', 'week', 'month')
            min_topic_size: Minimum size of topics
            min_post_length: Minimum number of tokens for a post to be considered
            extra_stopwords: Additional stopwords to remove
            n_neighbors: UMAP n_neighbors parameter
            n_components: UMAP n_components parameter
            min_cluster_size: HDBSCAN min_cluster_size parameter
        """
        self.language_model = language_model
        self.time_bin = time_bin
        self.min_topic_size = min_topic_size
        self.min_post_length = min_post_length
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_cluster_size = min_cluster_size
        
        # Initialize NLP components
        self.stop_words = set(stopwords.words('english'))
        if extra_stopwords:
            self.stop_words.update(extra_stopwords)
        
        # Will be populated after processing
        self.data = None
        self.documents = None
        self.doc_timestamps = None
        self.topic_model = None
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
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.jsonl'):
            data = []
            with open(file_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported file format. Please provide CSV or JSONL.")
        
        # Ensure required columns exist
        required_cols = ['user_id', 'timestamp', 'post_content']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by user_id and timestamp
        df = df.sort_values(['user_id', 'timestamp'])
        
        self.data = df
        return df
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text: lowercase, remove stopwords, URLs, emojis.
        For BERTopic, we keep the text structure rather than just returning tokens.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize to remove stopwords
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        
        # Rejoin into text
        processed_text = " ".join(tokens)
        
        return processed_text
    
    def preprocess_data(self, combine_by_window: bool = False) -> Tuple[List[str], pd.DataFrame]:
        """
        Preprocess all posts in the dataset.
        
        Args:
            combine_by_window: Whether to combine posts into time windows per user
            
        Returns:
            Tuple of (list of preprocessed documents, metadata DataFrame)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Preprocess each post
        df['processed_text'] = df['post_content'].apply(self.preprocess_text)
        
        # Filter out posts that are too short
        df = df[df['processed_text'].apply(lambda x: len(x.split())) >= self.min_post_length]
        
        if combine_by_window:
            # Add time bin column
            if self.time_bin == 'day':
                df['time_bin'] = df['timestamp'].dt.date
            elif self.time_bin == 'week':
                df['time_bin'] = df['timestamp'].dt.to_period('W').apply(lambda x: x.start_time)
            elif self.time_bin == 'month':
                df['time_bin'] = df['timestamp'].dt.to_period('M').apply(lambda x: x.start_time)
            
            # Combine posts by user and time bin
            grouped = df.groupby(['user_id', 'time_bin'])
            
            documents = []
            doc_meta = []
            
            for (user_id, time_bin), group in grouped:
                combined_text = " ".join(group['processed_text'])
                if combined_text.strip():  # Only add if there is text
                    documents.append(combined_text)
                    doc_meta.append({
                        'user_id': user_id,
                        'time_bin': time_bin,
                        'doc_id': len(documents) - 1,
                        'timestamp': group['timestamp'].mean()  # Use mean timestamp for the bin
                    })
            
            # Create metadata DataFrame
            doc_meta_df = pd.DataFrame(doc_meta)
            
        else:
            # Each post is a separate document
            documents = df['processed_text'].tolist()
            doc_meta_df = df[['user_id', 'timestamp']].copy()
            doc_meta_df['doc_id'] = doc_meta_df.index
        
        self.documents = documents
        self.doc_meta = doc_meta_df
        self.doc_timestamps = doc_meta_df['timestamp'].tolist()
        
        return documents, doc_meta_df
    
    def create_topic_model(self, documents: List[str], seed: int = 42) -> BERTopic:
        """
        Create and fit BERTopic model.
        
        Args:
            documents: List of preprocessed documents
            seed: Random seed for reproducibility
            
        Returns:
            Fitted BERTopic model
        """
        # Initialize sentence transformer
        print(f"Loading language model: {self.language_model}")
        embedding_model = SentenceTransformer(self.language_model)
        
        # Initialize UMAP for dimensionality reduction
        umap_model = UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=0.0,
            metric='cosine',
            random_state=seed
        )
        
        # Initialize HDBSCAN for clustering
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        
        # Initialize CountVectorizer
        vectorizer_model = CountVectorizer(
            stop_words="english",
            min_df=2,
            max_df=0.95
        )
        
        # Initialize BERTopic
        topic_model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ClassTfidfTransformer(),
            min_topic_size=self.min_topic_size,
            verbose=True
        )
        
        # Fit model
        print("Fitting BERTopic model...")
        self.topic_model = topic_model.fit(documents)
        
        return self.topic_model
    
    def get_document_topics(self) -> np.ndarray:
        """
        Get topic distribution for each document.
        
        Returns:
            numpy array of shape (n_documents, n_topics)
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created. Call create_topic_model() first.")
        
        # Get document-topic probabilities
        topic_distr, _ = self.topic_model.approximate_distribution(self.documents)
        
        # Store for later use
        self.doc_topics = topic_distr
        
        return topic_distr
    
    def calculate_user_topic_distributions(self) -> Dict[str, np.ndarray]:
        """
        Calculate aggregated topic distribution for each user.
        
        Returns:
            Dictionary mapping user_id to their topic distribution
        """
        if self.doc_topics is None:
            self.get_document_topics()
        
        user_topic_distributions = {}
        
        for user_id in self.doc_meta['user_id'].unique():
            # Get indices of documents for this user
            user_doc_indices = self.doc_meta[self.doc_meta['user_id'] == user_id]['doc_id'].values
            
            # Get topic distributions for these documents
            user_docs_topics = self.doc_topics[user_doc_indices]
            
            # Aggregate topic distributions (mean)
            user_topic_dist = user_docs_topics.mean(axis=0)
            
            user_topic_distributions[user_id] = user_topic_dist
        
        self.user_topic_distributions = user_topic_distributions
        return user_topic_distributions
    
    def calculate_gini_coefficient(self, array: np.ndarray) -> float:
        """
        Calculate the Gini coefficient for a distribution.
        
        Args:
            array: Distribution to calculate Gini coefficient for
            
        Returns:
            Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        """
        # Sort array
        array = np.sort(array)
        
        # Calculate Gini coefficient
        n = len(array)
        index = np.arange(1, n + 1)
        return (np.sum((2 * index - n - 1) * array)) / (n * np.sum(array))
    
    def calculate_shannon_entropy(self, array: np.ndarray) -> float:
        """
        Calculate Shannon entropy for a distribution.
        
        Args:
            array: Distribution to calculate entropy for
            
        Returns:
            Shannon entropy (higher = more diverse)
        """
        # Normalize if not already
        if np.sum(array) != 1:
            array = array / np.sum(array)
        
        # Calculate entropy
        return entropy(array)
    
    def calculate_top_n_ratio(self, array: np.ndarray, n: int = 2) -> float:
        """
        Calculate the ratio of the top N topics in a distribution.
        
        Args:
            array: Distribution to calculate ratio for
            n: Number of top topics to consider
            
        Returns:
            Ratio of top N topics (0-1)
        """
        # Sort in descending order
        sorted_array = np.sort(array)[::-1]
        
        # Calculate ratio
        return np.sum(sorted_array[:n]) / np.sum(sorted_array)
    
    def calculate_user_narrowness_scores(self) -> pd.DataFrame:
        """
        Calculate narrowness scores for each user using multiple metrics.
        
        Returns:
            DataFrame with user narrowness scores
        """
        if self.user_topic_distributions is None:
            self.calculate_user_topic_distributions()
        
        narrowness_data = []
        
        for user_id, topic_dist in self.user_topic_distributions.items():
            # Count posts by this user
            post_count = len(self.doc_meta[self.doc_meta['user_id'] == user_id])
            
            # Calculate metrics
            gini = self.calculate_gini_coefficient(topic_dist)
            entropy_score = self.calculate_shannon_entropy(topic_dist)
            top1_ratio = self.calculate_top_n_ratio(topic_dist, n=1)
            top2_ratio = self.calculate_top_n_ratio(topic_dist, n=2)
            
            # Find dominant topic
            dominant_topic = np.argmax(topic_dist)
            
            narrowness_data.append({
                'user_id': user_id,
                'post_count': post_count,
                'gini_coefficient': gini,
                'shannon_entropy': entropy_score,
                'top1_ratio': top1_ratio,
                'top2_ratio': top2_ratio,
                'dominant_topic': dominant_topic
            })
        
        # Create DataFrame
        narrowness_df = pd.DataFrame(narrowness_data)
        
        # Sort by Gini coefficient (descending)
        narrowness_df = narrowness_df.sort_values('gini_coefficient', ascending=False)
        
        self.user_narrowness_scores = narrowness_df
        return narrowness_df
    
    def calculate_temporal_topic_data(self) -> Dict[str, pd.DataFrame]:
        """
        Calculate topic proportions over time for each user.
        
        Returns:
            Dictionary mapping user_id to DataFrame with temporal topic data
        """
        if self.doc_topics is None:
            self.get_document_topics()
        
        temporal_data = {}
        
        for user_id in self.doc_meta['user_id'].unique():
            # Get data for this user
            user_docs = self.doc_meta[self.doc_meta['user_id'] == user_id].copy()
            
            # Add topic distributions
            user_topic_dists = self.doc_topics[user_docs['doc_id'].values]
            
            # Create a DataFrame for this user's temporal data
            if 'time_bin' in user_docs.columns:
                # If we combined by time window
                time_col = 'time_bin'
            else:
                # If each post is separate, bin by time
                user_docs['time_bin'] = pd.to_datetime(user_docs['timestamp'])
                if self.time_bin == 'day':
                    user_docs['time_bin'] = user_docs['time_bin'].dt.date
                elif self.time_bin == 'week':
                    user_docs['time_bin'] = user_docs['time_bin'].dt.to_period('W').apply(lambda x: x.start_time)
                elif self.time_bin == 'month':
                    user_docs['time_bin'] = user_docs['time_bin'].dt.to_period('M').apply(lambda x: x.start_time)
                time_col = 'time_bin'
            
            # Group by time bin and calculate mean topic distribution
            user_temporal = {}
            for time_bin in user_docs[time_col].unique():
                bin_indices = user_docs[user_docs[time_col] == time_bin].index
                if len(bin_indices) > 0:
                    rel_indices = [list(user_docs.index).index(i) for i in bin_indices]
                    bin_topic_dist = user_topic_dists[rel_indices]
                    user_temporal[time_bin] = bin_topic_dist.mean(axis=0)
            
            # Convert to DataFrame
            if user_temporal:
                user_temporal_df = pd.DataFrame(user_temporal).T
                num_topics = user_temporal_df.shape[1]
                user_temporal_df.columns = [f'Topic_{i}' for i in range(num_topics)]
                user_temporal_df.index.name = 'time_bin'
                user_temporal_df = user_temporal_df.sort_index()
                
                temporal_data[user_id] = user_temporal_df
        
        self.temporal_topic_data = temporal_data
        return temporal_data
    
    def detect_suspicious_users(self, post_frequency_threshold: float = 0.95, 
                               narrowness_threshold: float = 0.95,
                               similarity_threshold: float = 0.8) -> pd.DataFrame:
        """
        Flag suspicious users based on posting patterns and topic narrowness.
        
        Args:
            post_frequency_threshold: Percentile threshold for post frequency
            narrowness_threshold: Percentile threshold for topic narrowness (Gini coefficient)
            similarity_threshold: Jaccard similarity threshold for detecting duplicate posts
            
        Returns:
            DataFrame with flagged suspicious users and their metrics
        """
        if self.user_narrowness_scores is None:
            self.calculate_user_narrowness_scores()
        
        # Calculate post frequency thresholds
        post_freq_cutoff = np.percentile(self.user_narrowness_scores['post_count'], post_frequency_threshold)
        narrowness_cutoff = np.percentile(self.user_narrowness_scores['gini_coefficient'], narrowness_threshold)
        
        # Flag suspicious users
        suspicious = self.user_narrowness_scores.copy()
        suspicious['suspicious_post_freq'] = suspicious['post_count'] >= post_freq_cutoff
        suspicious['suspicious_narrowness'] = suspicious['gini_coefficient'] >= narrowness_cutoff
        
        # Calculate similarity between posts (this can be computationally expensive)
        duplicate_post_ratios = {}
        
        for user_id in suspicious['user_id']:
            user_posts = self.data[self.data['user_id'] == user_id]['post_content'].tolist()
            
            if len(user_posts) <= 1:
                duplicate_post_ratios[user_id] = 0
                continue
            
            # Calculate Jaccard similarity for pairs of posts
            similar_pairs = 0
            total_pairs = 0
            
            for i in range(len(user_posts)):
                for j in range(i+1, len(user_posts)):
                    total_pairs += 1
                    
                    # Tokenize posts
                    tokens1 = set(self.preprocess_text(user_posts[i]).split())
                    tokens2 = set(self.preprocess_text(user_posts[j]).split())
                    
                    if not tokens1 or not tokens2:
                        continue
                    
                    # Calculate Jaccard similarity
                    jaccard = len(tokens1.intersection(tokens2)) / len(tokens1.union(tokens2))
                    
                    if jaccard > similarity_threshold:
                        similar_pairs += 1
            
            duplicate_ratio = similar_pairs / max(1, total_pairs)
            duplicate_post_ratios[user_id] = duplicate_ratio
        
        suspicious['duplicate_post_ratio'] = suspicious['user_id'].map(duplicate_post_ratios)
        suspicious['suspicious_duplicates'] = suspicious['duplicate_post_ratio'] > 0.5
        
        # Overall suspicion flag
        suspicious['suspicious'] = (suspicious['suspicious_post_freq'] & 
                                   suspicious['suspicious_narrowness']) | suspicious['suspicious_duplicates']
        
        # Sort by suspiciousness
        suspicious = suspicious.sort_values(['suspicious', 'gini_coefficient', 'post_count'], 
                                           ascending=[False, False, False])
        
        return suspicious
    
    def visualize_topic_words(self, n_words: int = 10, save_path: Optional[str] = None):
        """
        Visualize top words for each topic using word clouds.
        
        Args:
            n_words: Number of words to show per topic
            save_path: Path to save the visualization
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created. Call create_topic_model() first.")
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Filter out the -1 topic (outliers)
        topic_info = topic_info[topic_info["Topic"] != -1]
        
        # Set up the figure
        n_topics = len(topic_info)
        n_cols = min(5, n_topics)
        n_rows = (n_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
        
        # Create word clouds for each topic
        for i, (_, row) in enumerate(topic_info.iterrows()):
            if i >= len(axes):
                break
                
            topic_id = row["Topic"]
            
            # Get top words for this topic
            words = self.topic_model.get_topic(topic_id)
            top_words = dict(words[:n_words])
            
            # Create word cloud
            wc = WordCloud(width=400, height=400, background_color='white', 
                           max_words=n_words, colormap='viridis')
            wc.generate_from_frequencies(top_words)
            
            # Plot
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f'Topic {topic_id}')
            axes[i].axis('off')
        
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def visualize_user_topic_pie(self, user_id: str, save_path: Optional[str] = None):
        """
        Visualize topic distribution for a user as a pie chart.
        
        Args:
            user_id: ID of the user to visualize
            save_path: Path to save the visualization
        """
        if self.user_topic_distributions is None:
            self.calculate_user_topic_distributions()
        
        if user_id not in self.user_topic_distributions:
            raise ValueError(f"User {user_id} not found in data.")
        
        # Get topic distribution for this user
        topic_dist = self.user_topic_distributions[user_id]
        
        # Only include topics with significant contribution
        threshold = 0.03
        significant_topics = [(i, prob) for i, prob in enumerate(topic_dist) if prob > threshold]
        other_prob = sum(prob for i, prob in enumerate(topic_dist) if prob <= threshold)
        
        # Sort by probability
        significant_topics.sort(key=lambda x: x[1], reverse=True)
        
        # Create labels and values
        labels = [f'Topic {i} ({prob:.2f})' for i, prob in significant_topics]
        if other_prob > 0:
            labels.append(f'Other ({other_prob:.2f})')
            
        values = [prob for _, prob in significant_topics]
        if other_prob > 0:
            values.append(other_prob)
        
        # Create pie chart
        plt.figure(figsize=(10, 10))
        plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab20.colors)
        plt.axis('equal')
        plt.title(f'Topic Distribution for User {user_id}')
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def visualize_user_temporal_topics(self, user_id: str, save_path: Optional[str] = None):
        """
        Visualize temporal topic distribution for a user as a stacked area chart.
        
        Args:
            user_id: ID of the user to visualize
            save_path: Path to save the visualization
        """
        if self.temporal_topic_data is None:
            self.calculate_temporal_topic_data()
        
        if user_id not in self.temporal_topic_data:
            raise ValueError(f"User {user_id} not found in temporal data.")
        
        # Get temporal data for this user
        user_temporal = self.temporal_topic_data[user_id]
        
        # Only include top topics
        top_n = min(5, user_temporal.shape[1])
        avg_dist = user_temporal.mean()
        top_topics = avg_dist.nlargest(top_n).index.tolist()
        
        # Create 'Other' category for remaining topics
        user_temporal['Other'] = user_temporal.drop(columns=top_topics).sum(axis=1)
        
        # Plot stacked area chart
        plt.figure(figsize=(12, 6))
        user_temporal[top_topics + ['Other']].plot.area(stacked=True, alpha=0.7, figsize=(12, 6))
        
        plt.title(f'Topic Evolution Over Time for User {user_id}')
        plt.xlabel('Time')
        plt.ylabel('Topic Proportion')
        plt.legend(title='Topics')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def visualize_narrowness_vs_frequency(self, metric: str = 'gini_coefficient', 
                                         save_path: Optional[str] = None):
        """
        Visualize user narrowness vs. post frequency.
        
        Args:
            metric: Narrowness metric to use ('gini_coefficient', 'shannon_entropy', 'top1_ratio', 'top2_ratio')
            save_path: Path to save the visualization
        """
        if self.user_narrowness_scores is None:
            self.calculate_user_narrowness_scores()
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Color by suspiciousness if available
        if 'suspicious' in self.user_narrowness_scores.columns:
            colors = ['red' if s else 'blue' for s in self.user_narrowness_scores['suspicious']]
            suspicious_count = sum(self.user_narrowness_scores['suspicious'])
            plt.scatter(self.user_narrowness_scores['post_count'], 
                       self.user_narrowness_scores[metric],
                       c=colors, alpha=0.7)
            plt.title(f'User Topic Narrowness vs. Post Frequency ({suspicious_count} suspicious users)')
        else:
            plt.scatter(self.user_narrowness_scores['post_count'], 
                       self.user_narrowness_scores[metric],
                       alpha=0.7)
            plt.title('User Topic Narrowness vs. Post Frequency')
            
        plt.xlabel('Number of Posts')
        plt.ylabel(f'Topic Narrowness ({metric})')
        plt.grid(True, alpha=0.3)
        
        # Add log scale for post count if range is large
        if max(self.user_narrowness_scores['post_count']) / min(self.user_narrowness_scores['post_count']) > 100:
            plt.xscale('log')
            
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
    def visualize_topic_embedding(self, save_path: Optional[str] = None):
        """
        Visualize topic embeddings using the model's own projection.
        BERTopic already does dimensionality reduction internally.
        
        Args:
            save_path: Path to save the visualization
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created. Call create_topic_model() first.")
        
        # Get topic info
        topic_info = self.topic_model.get_topic_info()
        
        # Filter out the -1 topic (outliers)
        filtered_topics = topic_info[topic_info["Topic"] != -1]["Topic"].values
        
        # Visualize topics
        fig = self.topic_model.visualize_topics(filtered_topics)
        
        if save_path:
            fig.write_image(save_path)
            
        fig.show()
    
    def export_user_topic_data(self, output_path: str):
        """
        Export user topic distribution and narrowness scores to CSV.
        
        Args:
            output_path: Path to save the CSV file
        """
        if self.user_narrowness_scores is None:
            self.calculate_user_narrowness_scores()
        
        # Create a copy of the narrowness DataFrame
        export_df = self.user_narrowness_scores.copy()
        
        # Add full topic distribution for each user
        n_topics = len(next(iter(self.user_topic_distributions.values())))
        for i in range(n_topics):
            export_df[f'Topic_{i}'] = export_df['user_id'].apply(
                lambda uid: self.user_topic_distributions[uid][i] if uid in self.user_topic_distributions else 0)
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        print(f"Exported user topic data to {output_path}")
    
    def export_topic_data(self, output_dir: str):
        """
        Export topic data for further analysis.
        
        Args:
            output_dir: Directory to save the outputs
        """
        if self.topic_model is None:
            raise ValueError("Topic model not created. Call create_topic_model() first.")
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Export topic info
        topic_info = self.topic_model.get_topic_info()
        topic_info.to_csv(f"{output_dir}/topic_info.csv", index=False)
        
        # Export top words per topic
        topic_words = []
        
        for topic_id in topic_info[topic_info["Topic"] != -1]["Topic"]:
            words = self.topic_model.get_topic(topic_id)
            for i, (word, weight) in enumerate(words[:20]):  # Export top 20 words
                topic_words.append({
                    'topic_id': topic_id,
                    'word': word,
                    'weight': weight,
                    'rank': i + 1
                })
        
        topic_words_df = pd.DataFrame(topic_words)
        topic_words_df.to_csv(f"{output_dir}/topic_words.csv", index=False)
        
        # Export topic similarity matrix
        topic_sims = self.topic_model.topic_embeddings_
        np.save(f"{output_dir}/topic_embeddings.npy", topic_sims)
        
        # Export temporal data if available
        if self.temporal_topic_data:
            import pickle
            with open(f"{output_dir}/temporal_topic_data.pkl", 'wb') as f:
                pickle.dump(self.temporal_topic_data, f)
        
        print(f"Exported topic data to {output_dir}")
    
    def run_full_pipeline(self, 
                         input_file: str, 
                         output_dir: str,
                         combine_by_window: bool = True,
                         visualize_topics: bool = True,
                         detect_suspicious: bool = True):
        """
        Run the complete analysis pipeline.
        
        Args:
            input_file: Path to input data file
            output_dir: Directory to save outputs
            combine_by_window: Whether to combine posts into time windows
            visualize_topics: Whether to create and save visualizations
            detect_suspicious: Whether to detect suspicious users
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Loading data...")
        self.load_data(input_file)
        
        print("2. Preprocessing data...")
        documents, _ = self.preprocess_data(combine_by_window=combine_by_window)
        
        print("3. Creating BERTopic model...")
        self.create_topic_model(documents)
        
        print("4. Calculating document topics...")
        self.get_document_topics()
        
        print("5. Calculating user topic distributions...")
        self.calculate_user_topic_distributions()
        
        print("6. Calculating user narrowness scores...")
        self.calculate_user_narrowness_scores()
        
        print("7. Calculating temporal topic data...")
        self.calculate_temporal_topic_data()
        
        # Export topic data
        print("8. Exporting topic data...")
        self.export_topic_data(output_dir)
        
        if detect_suspicious:
            print("9. Detecting suspicious users...")
            suspicious_users = self.detect_suspicious_users()
            suspicious_users.to_csv(f"{output_dir}/suspicious_users.csv", index=False)
            print(f"   Identified {sum(suspicious_users['suspicious'])} suspicious users")
        
        # Export user topic data
        self.export_user_topic_data(f"{output_dir}/user_topic_data.csv")
        
        if visualize_topics:
            print("10. Creating visualizations...")
            # Topic word clouds
            self.visualize_topic_words(save_path=f"{output_dir}/topic_words.png")
            
            # User narrowness vs frequency
            self.visualize_narrowness_vs_frequency(save_path=f"{output_dir}/narrowness_vs_frequency.png")
            
            # Topic embedding
            self.visualize_topic_embedding(save_path=f"{output_dir}/topic_embedding.png")
            
            # Create visualizations for top users
            top_users = self.user_narrowness_scores.head(5)['user_id'].tolist()
            for user_id in top_users:
                self.visualize_user_topic_pie(user_id, save_path=f"{output_dir}/user_{user_id}_topics.png")
                self.visualize_user_temporal_topics(user_id, save_path=f"{output_dir}/user_{user_id}_temporal.png")
        
        print("Pipeline completed successfully!")
        return {
            'num_documents': len(documents),
            'num_users': len(self.user_topic_distributions),
            'num_topics': self.topic_model.get_topic_info()[self.topic_model.get_topic_info()["Topic"] != -1].shape[0],
            'output_dir': output_dir
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    topic_system = BERTopicModelingSystem(
        language_model='all-MiniLM-L6-v2',  # Lightweight model, use 'all-mpnet-base-v2' for better quality
        time_bin='week',
        min_topic_size=10,
        extra_stopwords=['rt', 'http', 'https', 'amp'],
        n_neighbors=15,
        n_components=5,
        min_cluster_size=15
    )
    
    # Run the pipeline
    topic_system.run_full_pipeline(
        input_file="user_posts.csv",  # Replace with your data file
        output_dir="bertopic_analysis_results",
        combine_by_window=True,
        visualize_topics=True,
        detect_suspicious=True
    )
