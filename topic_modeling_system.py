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
from gensim.models.coherencemodel import CoherenceModel

# Dimensionality reduction for visualization
import umap
from sklearn.manifold import TSNE

# For measuring inequality/concentration
from scipy.stats import entropy
import matplotlib.cm as cm
from wordcloud import WordCloud

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')


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
        self.data = None
        self.dictionary = None
        self.corpus = None
        self.lda_model = None
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
    
    def preprocess_text(self, text: str) -> List[str]:
        """
        Preprocess text: lowercase, remove stopwords, URLs, emojis, and tokenize.
        
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
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', '', text)
        
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
        Preprocess all posts in the dataset.
        
        Args:
            combine_by_window: Whether to combine posts into time windows per user
            
        Returns:
            List of preprocessed documents (each document is a list of tokens)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Create a copy of the data
        df = self.data.copy()
        
        # Preprocess each post
        df['tokens'] = df['post_content'].apply(self.preprocess_text)
        
        # Filter out posts that are too short
        df = df[df['tokens'].apply(len) >= self.min_post_length]
        
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
                combined_tokens = [token for tokens_list in group['tokens'] for token in tokens_list]
                if combined_tokens:  # Only add if there are tokens
                    documents.append(combined_tokens)
                    doc_meta.append({
                        'user_id': user_id,
                        'time_bin': time_bin,
                        'doc_id': len(documents) - 1
                    })
            
            # Create metadata DataFrame
            self.doc_meta = pd.DataFrame(doc_meta)
            
        else:
            # Each post is a separate document
            documents = df['tokens'].tolist()
            self.doc_meta = df[['user_id', 'timestamp']].copy()
            self.doc_meta['doc_id'] = self.doc_meta.index
        
        return documents
    
    def create_lda_model(self, documents: List[List[str]], passes: int = 20) -> LdaModel:
        """
        Create and train an LDA topic model.
        
        Args:
            documents: List of preprocessed documents (lists of tokens)
            passes: Number of passes through the corpus during training
            
        Returns:
            Trained LDA model
        """
        # Create dictionary
        self.dictionary = corpora.Dictionary(documents)
        
        # Filter extremes (optional)
        self.dictionary.filter_extremes(no_below=5, no_above=0.7)
        
        # Create document-term matrix
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        
        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=passes,
            alpha='auto',
            eta='auto',
            random_state=42
        )
        
        return self.lda_model
    
    def get_document_topics(self) -> np.ndarray:
        """
        Get topic distribution for each document.
        
        Returns:
            numpy array of shape (n_documents, n_topics)
        """
        if self.lda_model is None or self.corpus is None:
            raise ValueError("LDA model not created. Call create_lda_model() first.")
        
        # Get topic distribution for each document
        doc_topics = np.zeros((len(self.corpus), self.num_topics))
        
        for i, doc_topic in enumerate(self.lda_model[self.corpus]):
            for topic_idx, prob in doc_topic:
                doc_topics[i, topic_idx] = prob
        
        self.doc_topics = doc_topics
        return doc_topics
    
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
                bin_topic_dist = user_topic_dists[bin_indices - min(user_docs.index)]
                user_temporal[time_bin] = bin_topic_dist.mean(axis=0)
            
            # Convert to DataFrame
            user_temporal_df = pd.DataFrame(user_temporal).T
            user_temporal_df.columns = [f'Topic_{i}' for i in range(self.num_topics)]
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
                    tokens1 = set(self.preprocess_text(user_posts[i]))
                    tokens2 = set(self.preprocess_text(user_posts[j]))
                    
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
        if self.lda_model is None:
            raise ValueError("LDA model not created. Call create_lda_model() first.")
        
        # Set up the figure
        n_cols = 5
        n_rows = (self.num_topics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows))
        axes = axes.flatten()
        
        # Create word clouds for each topic
        for i, topic in enumerate(range(self.num_topics)):
            if i >= len(axes):
                break
                
            # Get top words and their weights
            top_words = dict(self.lda_model.show_topic(topic, n_words))
            
            # Create word cloud
            wc = WordCloud(width=400, height=400, background_color='white', 
                           max_words=n_words, colormap='viridis')
            wc.generate_from_frequencies(top_words)
            
            # Plot
            axes[i].imshow(wc, interpolation='bilinear')
            axes[i].set_title(f'Topic {topic}')
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
        top_n = 5
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
    
    def visualize_topic_embedding(self, method: str = 'tsne', save_path: Optional[str] = None):
        """
        Visualize topic embeddings using dimensionality reduction.
        
        Args:
            method: Dimensionality reduction method ('tsne' or 'umap')
            save_path: Path to save the visualization
        """
        if self.doc_topics is None:
            self.get_document_topics()
        
        # Apply dimensionality reduction
        if method.lower() == 'tsne':
            reducer = TSNE(n_components=2, random_state=42)
        elif method.lower() == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError("Method must be 'tsne' or 'umap'")
        
        # Get document embeddings
        doc_topics_embedded = reducer.fit_transform(self.doc_topics)
        
        # Get dominant topic for each document
        dominant_topics = np.argmax(self.doc_topics, axis=1)
        
        # Create scatter plot
        plt.figure(figsize=(12, 10))
        
        # Color by topic
        scatter = plt.scatter(doc_topics_embedded[:, 0], doc_topics_embedded[:, 1], 
                             c=dominant_topics, cmap='tab20', alpha=0.7, s=30)
        
        plt.title(f'Document Topic Embedding ({method.upper()})')
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.grid(True, alpha=0.3)
        
        # Add legend for topics
        legend1 = plt.legend(*scatter.legend_elements(), title="Topics")
        plt.gca().add_artist(legend1)
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()
    
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
        for i in range(self.num_topics):
            export_df[f'Topic_{i}'] = export_df['user_id'].apply(
                lambda uid: self.user_topic_distributions[uid][i])
        
        # Export to CSV
        export_df.to_csv(output_path, index=False)
        print(f"Exported user topic data to {output_path}")
    
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
        documents = self.preprocess_data(combine_by_window=combine_by_window)
        
        print(f"3. Creating LDA model with {self.num_topics} topics...")
        self.create_lda_model(documents)
        
        print("4. Calculating document topics...")
        self.get_document_topics()
        
        print("5. Calculating user topic distributions...")
        self.calculate_user_topic_distributions()
        
        print("6. Calculating user narrowness scores...")
        self.calculate_user_narrowness_scores()
        
        print("7. Calculating temporal topic data...")
        self.calculate_temporal_topic_data()
        
        if detect_suspicious:
            print("8. Detecting suspicious users...")
            suspicious_users = self.detect_suspicious_users()
            suspicious_users.to_csv(f"{output_dir}/suspicious_users.csv", index=False)
            print(f"   Identified {sum(suspicious_users['suspicious'])} suspicious users")
        
        # Export user topic data
        self.export_user_topic_data(f"{output_dir}/user_topic_data.csv")
        
        if visualize_topics:
            print("9. Creating visualizations...")
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
            'num_topics': self.num_topics,
            'output_dir': output_dir
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    topic_system = UserTopicModelingSystem(
        num_topics=20,
        time_bin='week',
        lemmatize=True,
        extra_stopwords=['rt', 'http', 'https', 'amp']
    )
    
    # Run the pipeline
    topic_system.run_full_pipeline(
        input_file="user_posts.csv",  # Replace with your data file
        output_dir="topic_analysis_results",
        combine_by_window=True,
        visualize_topics=True,
        detect_suspicious=True
    )
