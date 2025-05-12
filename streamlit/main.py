# main.py
import streamlit as st

# 1. Call st.set_page_config() FIRST - This is the CRITICAL FIX
st.set_page_config(layout="wide", page_title="Social Media Text Processor & Analyzer")

# 2. Then other imports
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import emoji
import nltk
# NLTK resource-dependent imports will be handled after the check
from gensim.models.coherencemodel import CoherenceModel
import json
import string
import os
from pathlib import Path
import tempfile
import shutil
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import pickle
import traceback
import io
import sys

# --- NLTK Resource Check and Download Function ---
@st.cache_resource 
def ensure_nltk_resources():
    resources_to_check = {
        "stopwords": ("corpora/stopwords", "stopwords"),
        "punkt": ("tokenizers/punkt", "punkt"),
        "wordnet": ("corpora/wordnet", "wordnet")
    }
    all_good = True
    messages = [] 
    for name, (path, package_id) in resources_to_check.items():
        try:
            nltk.data.find(path)
        except LookupError:
            messages.append(f"INFO: Downloading NLTK resource: {name} ({package_id})...")
            try:
                nltk.download(package_id, quiet=True)
                messages.append(f"SUCCESS: NLTK resource '{name}' downloaded.")
            except Exception as e:
                messages.append(f"ERROR: Failed to download NLTK resource '{name}': {e}")
                all_good = False
    
    st.session_state.nltk_messages_for_sidebar = messages
    st.session_state.nltk_resources_all_good = all_good
    if all_good and messages: 
         st.session_state.nltk_messages_for_sidebar.append("SUCCESS: All required NLTK resources checked/downloaded.")
    return all_good

# --- Call NLTK check early. It populates session_state but doesn't use st.* calls itself.
NLTK_READY = ensure_nltk_resources()

# --- Import UserTopicModelingSystem (conditionally) ---
USER_TOPIC_MODELING_AVAILABLE = False
USER_TOPIC_MODELING_ERROR_MESSAGE = ""
nltk_stopwords_global = None # Use a different name to avoid conflict

if NLTK_READY:
    try:
        from user_topic_modeling import UserTopicModelingSystem 
        from nltk.corpus import stopwords # Import here after NLTK_READY
        nltk_stopwords_global = stopwords # Assign to be used by preprocessing functions
        USER_TOPIC_MODELING_AVAILABLE = True
    except ImportError as e:
        USER_TOPIC_MODELING_ERROR_MESSAGE = (
            f"Could not import UserTopicModelingSystem. \n"
            f"1. Ensure 'user_topic_modeling.py' (the file containing the UserTopicModelingSystem class definition) "
            f"is in the same directory as this app ('{os.getcwd()}').\n"
            f"2. Ensure all its dependencies (gensim, numpy, etc.) are installed.\n"
            f"3. Check for circular imports (e.g., if 'user_topic_modeling.py' mistakenly tries to import this app file).\n"
            f"Error: {e}"
        )
    except Exception as e_other:
        USER_TOPIC_MODELING_ERROR_MESSAGE = (
            f"An unexpected error occurred while trying to import UserTopicModelingSystem. "
            f"This might be due to an issue within 'user_topic_modeling.py' itself (like an incorrect NLTK data path or a deeper dependency problem).\n"
            f"Error: {e_other}\n\nTraceback:\n{traceback.format_exc()}"
        )
else:
    USER_TOPIC_MODELING_ERROR_MESSAGE = "NLTK resources could not be verified/downloaded. Core functionalities will be unavailable."

# --- Now it's safe to use st.* commands for the rest of the app ---
st.title("📄 Social Media Text Processor & Topic Analyzer")

if not USER_TOPIC_MODELING_AVAILABLE and not st.session_state.get("error_shown_once_main_import", False):
    st.error(USER_TOPIC_MODELING_ERROR_MESSAGE)
    st.session_state.error_shown_once_main_import = True
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("1. Upload Data & Select Columns")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="file_uploader")

if 'nltk_messages_for_sidebar' in st.session_state and st.session_state.nltk_messages_for_sidebar:
    expanded_nltk_status = not st.session_state.get('nltk_resources_all_good', True) 
    with st.sidebar.expander("NLTK Resource Status", expanded=expanded_nltk_status): 
        for msg in st.session_state.nltk_messages_for_sidebar:
            if msg.startswith("ERROR:") : st.error(msg)
            elif msg.startswith("INFO:") : st.info(msg)
            else: st.success(msg)
    # Clear messages after displaying so they don't reappear on every interaction
    if 'nltk_messages_displayed_once' not in st.session_state:
        st.session_state.nltk_messages_for_sidebar = [] # Clear them
        st.session_state.nltk_messages_displayed_once = True


df_original = None 
if uploaded_file is not None:
    try:
        df_original = pd.read_csv(uploaded_file)
        st.sidebar.success("File Uploaded Successfully!")
        column_options = ["<Select a Column>"] + df_original.columns.tolist()
        user_id_col = st.sidebar.selectbox("Select User ID Column (Needed for Topic Modeling)", column_options, index=0, key="user_id_sel")
        text_col = st.sidebar.selectbox("Select Tweet/Text Column *", column_options, index=0, key="text_col_sel")
        datetime_col = st.sidebar.selectbox("Select Date/Time Column (Needed for Topic Modeling)", column_options, index=0, key="datetime_col_sel")
        st.session_state.user_id_col_name = user_id_col if user_id_col != "<Select a Column>" else None
        st.session_state.text_col_name = text_col if text_col != "<Select a Column>" else None
        st.session_state.datetime_col_name = datetime_col if datetime_col != "<Select a Column>" else None
        if text_col == "<Select a Column>":
            st.sidebar.warning("Please select the Tweet/Text column to process.")
        else:
            if df_original is not None:
                st.header("Original Data Preview (First 5 rows)") 
                st.dataframe(df_original.head())
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        df_original = None 

# --- Helper Class for Streamlit Output & Preprocessing Functions ---
class StreamlitLogHandler(io.StringIO):
    def __init__(self, placeholder):
        super().__init__(); self.placeholder = placeholder; self._buffer = [] 
    def write(self, s):
        if s.strip(): self._buffer.append(s); log_content = "".join(self._buffer); self.placeholder.code(log_content.strip(), language='text')
    def flush(self): pass

def remove_urls(text): return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
def remove_html(text): return BeautifulSoup(text, "html.parser").get_text()
def remove_special_chars(text): return re.sub(r'[^A-Za-z0-9\s.,!?:;\'-@#_]', '', text)
def remove_hashtags(text): return re.sub(r'#\w+', '', text)
def extract_hashtags(text): return re.findall(r'#\w+', text)
def remove_mentions(text): return re.sub(r'@\w+', '', text)
def extract_mentions(text): return re.findall(r'@\w+', text)
def remove_emojis_func(text): return emoji.replace_emoji(text, replace='')
def demojize_emojis_func(text): return emoji.demojize(text)
def convert_to_lowercase(text): return text.lower()
def remove_stopwords_nltk(text, custom_stopwords=None):
    if not nltk_stopwords_global: return str(text) # NLTK not ready
    stop_words_set = set(nltk_stopwords_global.words('english'))
    if custom_stopwords: stop_words_set.update(custom_stopwords)
    words = str(text).split(); return " ".join([word for word in words if word.lower() not in stop_words_set])
def remove_numbers(text): return re.sub(r'\d+', '', text)
def remove_punctuation(text): return str(text).translate(str.maketrans('', '', string.punctuation.replace('@','').replace('#','')))

# --- Dashboard Plotting Functions ---
# (Implementations from the previous complete main.py - keep them here)
def display_topic_words_dashboard(topic_id_sel, topics_df_local):
    if topics_df_local is None or topics_df_local.empty: st.warning("Topic words data not available."); return
    topic_words_data = topics_df_local[topics_df_local['topic_id'] == topic_id_sel]
    if topic_words_data.empty: st.warning(f"No words for Topic {topic_id_sel}."); return
    word_dict_cloud = dict(zip(topic_words_data['word'], topic_words_data['weight']))
    if not word_dict_cloud: st.warning(f"Word dictionary empty for Topic {topic_id_sel}."); return
    try:
        wc = WordCloud(width=800, height=300, background_color='white', colormap='viridis', max_words=30).generate_from_frequencies(word_dict_cloud)
        fig_wc, ax_wc = plt.subplots(figsize=(10, 4)); ax_wc.imshow(wc, interpolation='bilinear'); ax_wc.axis('off'); ax_wc.set_title(f'Word Cloud for Topic {topic_id_sel}')
        st.pyplot(fig_wc); plt.close(fig_wc)
    except Exception as e: st.error(f"Word cloud error: {e}")
    try:
        fig_bar, ax_bar = plt.subplots(figsize=(10, 4)); sns.barplot(x='weight', y='word', data=topic_words_data.sort_values('weight', ascending=False).head(15), ax=ax_bar, palette="viridis")
        ax_bar.set_title(f'Top 15 Words for Topic {topic_id_sel}'); plt.tight_layout(); st.pyplot(fig_bar); plt.close(fig_bar)
    except Exception as e: st.error(f"Bar chart error: {e}")

def plot_user_topics_dashboard(user_id_str, user_topic_df_local, suspicious_df_local, num_topics_dash):
    if user_topic_df_local is None or user_topic_df_local.empty: return
    if user_id_str not in user_topic_df_local['user_id'].astype(str).values: st.warning(f"User ID '{user_id_str}' not found."); return
    user_data = user_topic_df_local[user_topic_df_local['user_id'].astype(str) == user_id_str].iloc[0]
    topic_props = [user_data.get(f'Topic_{i}', 0) for i in range(num_topics_dash)]
    threshold = 0.03; significant_topics = [{'id': i, 'prop': p} for i, p in enumerate(topic_props) if p > threshold]
    other_prop = sum(p for i, p in enumerate(topic_props) if p <= threshold)
    significant_topics.sort(key=lambda x: x['prop'], reverse=True)
    labels = [f'Topic {item["id"]} ({item["prop"]:.2f})' for item in significant_topics]; values = [item['prop'] for item in significant_topics]
    if other_prop > 0.001: labels.append(f'Other ({other_prop:.2f})'); values.append(other_prop)
    if not values: st.info(f"User {user_id_str} has no significant topic distribution."); return
    fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='percent', hoverinfo='label+percent', hole=0.3)])
    fig_pie.update_layout(title_text=f'Topic Distribution for User {user_id_str}', height=400, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_pie, use_container_width=True)
    st.markdown(f"**Metrics for User: {user_id_str}**"); st.markdown(f"- Post count: `{user_data.get('post_count', 'N/A')}`")
    st.markdown(f"- Gini: `{user_data.get('gini_coefficient', float('nan')):.4f}` | Shannon: `{user_data.get('shannon_entropy', float('nan')):.4f}` | Top-1 Ratio: `{user_data.get('top1_ratio', float('nan')):.4f}`")
    if suspicious_df_local is not None and user_id_str in suspicious_df_local['user_id'].astype(str).values:
        susp_data = suspicious_df_local[suspicious_df_local['user_id'].astype(str) == user_id_str].iloc[0]
        if susp_data.get('suspicious', False):
            st.warning("⚠️ This user has been flagged as suspicious.")
            reasons = [r for r, c in [("high posting frequency", 'suspicious_post_freq'), ("narrow topic focus", 'suspicious_narrowness'), ("high duplicate content", 'suspicious_duplicates')] if susp_data.get(c, False)]
            if reasons: st.write(f"   Reasons: {', '.join(reasons)}")

def plot_topic_concentration_dashboard(metric_col, log_scale_flag, user_topic_df_local, suspicious_df_local):
    if user_topic_df_local is None or user_topic_df_local.empty: return
    if metric_col not in user_topic_df_local.columns: st.error(f"Metric '{metric_col}' not in data."); return
    fig_conc = px.scatter(user_topic_df_local, x='post_count', y=metric_col, hover_data=['user_id', 'post_count', 'gini_coefficient', 'shannon_entropy', 'top1_ratio'], color='dominant_topic', size='post_count', size_max=25, opacity=0.7, log_x=log_scale_flag, title=f'User Topic Concentration ({metric_col.replace("_", " ").title()}) vs Post Frequency')
    mean_val_conc = user_topic_df_local[metric_col].mean(); fig_conc.add_hline(y=mean_val_conc, line_dash="dash", line_color="red", annotation_text=f"Mean: {mean_val_conc:.3f}", annotation_position="bottom right")
    if suspicious_df_local is not None and 'suspicious' in suspicious_df_local.columns:
        susp_users_list = suspicious_df_local[suspicious_df_local['suspicious']]['user_id'].astype(str).tolist()
        susp_plot_df = user_topic_df_local[user_topic_df_local['user_id'].astype(str).isin(susp_users_list)]
        if not susp_plot_df.empty: fig_conc.add_trace(go.Scatter(x=susp_plot_df['post_count'], y=susp_plot_df[metric_col], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=15, line=dict(color='red', width=3), symbol='circle-open'), name='Suspicious Outline', text=susp_plot_df['user_id'], hoverinfo='text'))
    fig_conc.update_layout(xaxis_title="Number of Posts", yaxis_title=f"{metric_col.replace('_', ' ').title()}", height=500, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_conc, use_container_width=True)

def plot_suspicious_comparison_dashboard(metric_col_susp, user_topic_df_local, suspicious_df_local):
    if user_topic_df_local is None or suspicious_df_local is None: return
    user_topic_df_local['user_id'] = user_topic_df_local['user_id'].astype(str); suspicious_df_local['user_id'] = suspicious_df_local['user_id'].astype(str)
    merged_df = user_topic_df_local.merge(suspicious_df_local[['user_id', 'suspicious', 'duplicate_post_ratio']], on='user_id', how='left')
    merged_df['suspicious'] = merged_df['suspicious'].fillna(False); merged_df['user_type'] = merged_df['suspicious'].map({True: 'Suspicious', False: 'Normal'})
    if metric_col_susp == 'duplicate_post_ratio' and 'duplicate_post_ratio' not in merged_df.columns: st.error("'duplicate_post_ratio' column not found."); return
    if metric_col_susp not in merged_df.columns: st.error(f"Metric column '{metric_col_susp}' not found."); return
    fig_violin = px.violin(merged_df, x='user_type', y=metric_col_susp, color='user_type', box=True, points='all', hover_data=['user_id', 'post_count'], title=f'Distribution of {metric_col_susp.replace("_", " ").title()}: Suspicious vs Normal Users')
    fig_violin.update_layout(height=450, margin=dict(t=50, b=20, l=20, r=20)); st.plotly_chart(fig_violin, use_container_width=True)

def plot_topic_similarity_heatmap_dashboard(user_topic_df_local, num_topics_dash):
    if user_topic_df_local is None: return
    topic_cols_sim = [f'Topic_{i}' for i in range(num_topics_dash)]
    for col in topic_cols_sim:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0 
    topic_vectors_sim = user_topic_df_local[topic_cols_sim].T.values 
    if topic_vectors_sim.shape[0] == 0: st.warning("No topics for similarity matrix."); return
    similarity_matrix = cosine_similarity(topic_vectors_sim) 
    fig_heatmap = px.imshow(similarity_matrix, labels=dict(x="Topic", y="Topic", color="Similarity"), x=[f'T{i}' for i in range(num_topics_dash)], y=[f'T{i}' for i in range(num_topics_dash)], color_continuous_scale='Viridis', aspect="auto")
    fig_heatmap.update_layout(title='Topic Similarity Matrix (Cosine Similarity)', height=500, margin=dict(t=50, b=20, l=20, r=20)); st.plotly_chart(fig_heatmap, use_container_width=True)

def plot_topic_similarity_network_dashboard(user_topic_df_local, num_topics_dash, min_similarity_threshold):
    if user_topic_df_local is None: st.warning("User topic data not available for network plot."); return
    topic_cols_net = [f'Topic_{i}' for i in range(num_topics_dash)]; 
    for col in topic_cols_net:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0
    topic_vectors_net = user_topic_df_local[topic_cols_net].T.values
    if topic_vectors_net.shape[0] == 0: st.warning("No topics for network plot."); return
    similarity_matrix_net = cosine_similarity(topic_vectors_net)
    G = nx.Graph(); 
    for i in range(num_topics_dash): G.add_node(i, label=f"T{i}")
    for i in range(num_topics_dash):
        for j in range(i + 1, num_topics_dash):
            if similarity_matrix_net[i, j] >= min_similarity_threshold: G.add_edge(i, j, weight=similarity_matrix_net[i, j])
    if not G.edges(): st.info("No topic pairs meet the similarity threshold to draw a network."); return
    pos = nx.spring_layout(G, seed=42, k=0.7/np.sqrt(G.number_of_nodes()) if G.number_of_nodes() > 1 else 0.7) 
    edge_x, edge_y = [], []; 
    for edge in G.edges(): x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    node_x, node_y, node_text, node_size_val = [], [], [], []
    for node in G.nodes(): 
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(G.nodes[node]['label']); node_size_val.append(15 + G.degree(node) * 3)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=node_text, textposition="top center", hoverinfo='text', 
                            marker=dict(showscale=True, colorscale='YlGnBu', size=node_size_val, 
                                        colorbar=dict(thickness=15, title='Node Degree', xanchor='left', titleside='right'), 
                                        line_width=2, color=[G.degree(node) for node in G.nodes()]))
    fig_net = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(title='Topic Similarity Network', titlefont_size=16, showlegend=False, hovermode='closest', margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=500))
    st.plotly_chart(fig_net, use_container_width=True)

def plot_user_clustering_dashboard(n_clusters_viz, show_stats_flag, show_heatmap_flag, user_topic_df_local, suspicious_df_local, num_topics_dash):
    if user_topic_df_local is None: return
    topic_cols_cluster = [f'Topic_{i}' for i in range(num_topics_dash)]; 
    for col in topic_cols_cluster:
        if col not in user_topic_df_local.columns: user_topic_df_local[col] = 0
    X_cluster = user_topic_df_local[topic_cols_cluster].fillna(0).values
    if X_cluster.shape[0] < 2 or X_cluster.shape[0] < n_clusters_viz: st.warning(f"Not enough users ({X_cluster.shape[0]}) or too few for {n_clusters_viz} clusters."); return
    perplexity_val = min(30.0, max(5.0, X_cluster.shape[0] - 1.0)) 
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_val, n_iter=300, init='pca', learning_rate='auto')
        X_tsne = tsne.fit_transform(X_cluster)
    except Exception as e: st.error(f"t-SNE error: {e}"); return
    current_n_clusters = n_clusters_viz
    if X_cluster.shape[0] < n_clusters_viz : current_n_clusters = max(1, X_cluster.shape[0]); st.warning(f"Reduced clusters to {current_n_clusters}.")
    if current_n_clusters <= 1: st.error("Cannot perform clustering with less than 2 clusters."); return # Kmeans needs at least 2 clusters
    kmeans = KMeans(n_clusters=current_n_clusters, random_state=42, n_init='auto'); clusters = kmeans.fit_predict(X_cluster)
    plot_df = pd.DataFrame({'x': X_tsne[:, 0], 'y': X_tsne[:, 1], 'user_id': user_topic_df_local['user_id'], 'cluster': clusters.astype(str), 'post_count': user_topic_df_local['post_count'], 'gini': user_topic_df_local['gini_coefficient']})
    fig_scatter = px.scatter(plot_df, x='x', y='y', color='cluster', size='post_count', hover_data=['user_id', 'gini', 'post_count'], size_max=20, opacity=0.7, title=f'User Clustering (K-means, {current_n_clusters} clusters) with t-SNE')
    if suspicious_df_local is not None and 'suspicious' in suspicious_df_local.columns:
        susp_ids = suspicious_df_local[suspicious_df_local['suspicious']]['user_id'].astype(str).tolist()
        susp_plot_df_points = plot_df[plot_df['user_id'].astype(str).isin(susp_ids)]
        if not susp_plot_df_points.empty: fig_scatter.add_trace(go.Scatter(x=susp_plot_df_points['x'], y=susp_plot_df_points['y'], mode='markers', marker=dict(color='rgba(0,0,0,0)', size=18, line=dict(color='red', width=3), symbol='circle-open'), name='Suspicious Outline', hoverinfo='text', text=susp_plot_df_points['user_id']))
    fig_scatter.update_layout(xaxis_title="t-SNE Dimension 1", yaxis_title="t-SNE Dimension 2", height=600, margin=dict(t=50,b=20,l=20,r=20)); st.plotly_chart(fig_scatter, use_container_width=True)
    if show_stats_flag:
        cluster_stats = plot_df.groupby('cluster').agg(User_Count=('user_id', 'count'), Avg_Gini=('gini', 'mean'), Avg_Post_Count=('post_count', 'mean')).reset_index()
        st.write("Cluster Statistics:"); st.dataframe(cluster_stats.style.background_gradient(cmap='Blues'))
        if show_heatmap_flag:
            st.write("Dominant Topics per Cluster (Mean Weights):")
            cluster_topic_means = []
            for cid_str in sorted(plot_df['cluster'].unique()):
                user_ids_in_cluster = plot_df[plot_df['cluster'] == cid_str]['user_id']
                cluster_data = user_topic_df_local[user_topic_df_local['user_id'].isin(user_ids_in_cluster)]
                means = [cluster_data.get(f'Topic_{i}', pd.Series(dtype=float)).mean() for i in range(num_topics_dash)]
                cluster_topic_means.append(means)
            heatmap_df = pd.DataFrame(cluster_topic_means, columns=[f'T{i}' for i in range(num_topics_dash)], index=[f'Cluster {i}' for i in sorted(plot_df['cluster'].unique())])
            fig_heatmap_cluster = px.imshow(heatmap_df.fillna(0), labels=dict(x="Topic", y="Cluster", color="Mean Weight"), color_continuous_scale='Viridis', aspect="auto")
            fig_heatmap_cluster.update_layout(title='Cluster-Topic Distribution Heatmap', height=max(300, 40 * current_n_clusters), margin=dict(t=50,b=20,l=20,r=20)); st.plotly_chart(fig_heatmap_cluster, use_container_width=True)

def plot_temporal_topics_dashboard(user_id_str, show_individual_flag, temporal_data_dict, num_topics_dash):
    if not temporal_data_dict or user_id_str not in temporal_data_dict: st.info(f"No temporal data for user {user_id_str}."); return
    user_temporal_df = temporal_data_dict[user_id_str]
    if not isinstance(user_temporal_df, pd.DataFrame) or user_temporal_df.empty: st.info(f"Temporal data for {user_id_str} empty/invalid."); return
    try: user_temporal_df.index = pd.to_datetime(user_temporal_df.index)
    except: st.warning("Could not convert temporal index to datetime."); 
    top_n = 5; avg_dist = user_temporal_df.mean()
    existing_cols = [col for col in [f'Topic_{i}' for i in range(num_topics_dash)] if col in avg_dist.index]
    if not existing_cols: st.warning("No topic columns in temporal data."); return
    top_topics = avg_dist[existing_cols].nlargest(top_n).index.tolist()
    plot_df = user_temporal_df.copy()
    other_cols = [col for col in existing_cols if col not in top_topics]
    if other_cols: plot_df['Other_Topics'] = user_temporal_df[other_cols].sum(axis=1)
    fig_stack = go.Figure()
    for topic in top_topics:
        if topic in plot_df.columns: fig_stack.add_trace(go.Scatter(x=plot_df.index, y=plot_df[topic], mode='lines', stackgroup='one', name=topic))
    if 'Other_Topics' in plot_df.columns: fig_stack.add_trace(go.Scatter(x=plot_df.index, y=plot_df['Other_Topics'], mode='lines', stackgroup='one', name='Other Topics'))
    fig_stack.update_layout(title=f'Topic Evolution Over Time for User {user_id_str} (Stacked)', xaxis_title='Time', yaxis_title='Topic Proportion', hovermode='x unified', height=450, margin=dict(t=50,b=20,l=20,r=20))
    st.plotly_chart(fig_stack, use_container_width=True)
    if show_individual_flag:
        fig_ind = go.Figure()
        for topic in top_topics:
            if topic in user_temporal_df.columns: fig_ind.add_trace(go.Scatter(x=user_temporal_df.index, y=user_temporal_df[topic], mode='lines+markers', name=topic))
        fig_ind.update_layout(title=f'Individual Top Topic Evolution for User {user_id_str}', xaxis_title='Time', yaxis_title='Topic Proportion', height=450, margin=dict(t=50,b=20,l=20,r=20))
        st.plotly_chart(fig_ind, use_container_width=True)

# --- Main Dashboard Rendering Function ---
def render_detailed_dashboard(results_dir_path_str):
    st.header("🔬 Detailed Topic Analysis Dashboard")
    results_dir = Path(results_dir_path_str)
    user_topic_df, suspicious_df, topics_df, temporal_topic_data_loaded = None, None, None, None
    num_topics_dash = st.session_state.get("lda_num_topics_run", 10) 
    user_topic_df_path = results_dir / "user_topic_data.csv"
    if user_topic_df_path.exists(): user_topic_df = pd.read_csv(user_topic_df_path)
    else: st.error("user_topic_data.csv not found!"); return
    suspicious_df_path = results_dir / "suspicious_users.csv"
    if suspicious_df_path.exists(): suspicious_df = pd.read_csv(suspicious_df_path)
    topics_df_path = results_dir / "topic_words.csv"
    if topics_df_path.exists(): topics_df = pd.read_csv(topics_df_path)
    temporal_data_pickle_path = results_dir / "temporal_topic_data.pkl" 
    if temporal_data_pickle_path.exists():
        try:
            with open(temporal_data_pickle_path, 'rb') as f_pkl: temporal_topic_data_loaded = pickle.load(f_pkl)
        except Exception as e_pkl: st.warning(f"Could not load temporal_topic_data.pkl: {e_pkl}")
    
    tab_titles = ["Topic Words", "User Distributions", "Topic Concentration", "Suspicious Users", "Temporal Analysis", "Clustering & Similarity"]
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)
    with tab1:
        if topics_df is not None:
            selected_topic_id_dash = st.slider("Select Topic ID for words", 0, max(0, num_topics_dash - 1), 0, key="dash_topic_slider_words")
            if num_topics_dash > 0 : display_topic_words_dashboard(selected_topic_id_dash, topics_df)
    with tab2:
        if user_topic_df is not None:
            user_list_dash = sorted(user_topic_df['user_id'].astype(str).unique())
            if user_list_dash:
                selected_user_dash = st.selectbox("Select User for distribution", user_list_dash, index=0, key="dash_user_select_dist")
                plot_user_topics_dashboard(selected_user_dash, user_topic_df, suspicious_df, num_topics_dash)
    with tab3:
        if user_topic_df is not None:
            metric_opts = {'Gini Coefficient': 'gini_coefficient', 'Shannon Entropy': 'shannon_entropy', 'Top-1 Ratio': 'top1_ratio'}
            metric_disp = st.selectbox("Select Metric for concentration", list(metric_opts.keys()), key="dash_conc_metric_main")
            log_s = st.checkbox("Log Scale (Posts) for concentration", False, key="dash_conc_log_main")
            plot_topic_concentration_dashboard(metric_opts[metric_disp], log_s, user_topic_df, suspicious_df)
    with tab4:
        if suspicious_df is not None and user_topic_df is not None and 'suspicious' in suspicious_df.columns:
            if suspicious_df['suspicious'].sum() > 0:
                susp_users_display_dash = suspicious_df[suspicious_df['suspicious']]
                st.write("Top 10 Flagged Suspicious Users:"); st.dataframe(susp_users_display_dash[['user_id', 'post_count', 'gini_coefficient', 'top1_ratio', 'duplicate_post_ratio']].head(10).style.background_gradient(cmap='Reds'))
                metric_opts_susp = {'Gini Coefficient': 'gini_coefficient', 'Duplicate Post Ratio': 'duplicate_post_ratio'}
                metric_disp_susp = st.selectbox("Compare Metric for suspicious", list(metric_opts_susp.keys()), key="dash_susp_compare_metric_main")
                plot_suspicious_comparison_dashboard(metric_opts_susp[metric_disp_susp], user_topic_df, suspicious_df)
            else: st.info("No users were flagged as suspicious.")
    with tab5:
        if temporal_topic_data_loaded:
            user_list_temp = sorted([str(k) for k in temporal_topic_data_loaded.keys()])
            if user_list_temp:
                sel_user_temp = st.selectbox("Select User for temporal plot", user_list_temp, key="dash_temp_user_main")
                show_ind_temp = st.checkbox("Show Individual Topic Lines (Temporal)", False, key="dash_temp_individual_main_cb")
                plot_temporal_topics_dashboard(sel_user_temp, show_ind_temp, temporal_topic_data_loaded, num_topics_dash)
        else: st.warning("Temporal data file (temporal_topic_data.pkl) not found/loaded.")
    with tab6:
        col_sim, col_clust = st.columns(2)
        with col_sim:
            st.markdown("#### Topic Similarity"); 
            if user_topic_df is not None: 
                plot_topic_similarity_heatmap_dashboard(user_topic_df, num_topics_dash)
                min_sim_net = st.slider("Min Similarity for Network", 0.0, 1.0, 0.3, 0.05, key="dash_topic_net_sim_main")
                plot_topic_similarity_network_dashboard(user_topic_df, num_topics_dash, min_sim_net)
        with col_clust:
            st.markdown("#### User Clustering"); 
            if user_topic_df is not None:
                n_clust_dash = st.slider("Number of User Clusters (K-Means)", 2, 15, 5, key="dash_user_clust_num_main")
                show_stats_c = st.checkbox("Show Cluster Stats", True, key="dash_user_clust_stats_main_cb")
                show_hm_c = st.checkbox("Show Cluster-Topic Heatmap", True, key="dash_user_clust_hm_main_cb")
                plot_user_clustering_dashboard(n_clust_dash, show_stats_c, show_hm_c, user_topic_df, suspicious_df, num_topics_dash)
    st.markdown("---")
    if st.button("⬅️ Back to Main Setup & Results", key="back_to_main_from_dashboard_btn_main"):
        st.session_state.viewing_dashboard = False; st.session_state.show_detailed_dashboard_button = True
        st.experimental_rerun()

# --- Main App Flow (Cleaning, LDA Run Button, Dashboard Button) ---
if df_original is not None and st.session_state.get('text_col_name') is not None:
    if 'processed_df' not in st.session_state: # If cleaning not done yet
        # The "Process Data" button and its logic are defined under the cleaning options in the sidebar.
        # This main area will show results after "Process Data" is clicked.
        pass
    
    if USER_TOPIC_MODELING_AVAILABLE and 'processed_df' in st.session_state and st.session_state.processed_df is not None:
        # LDA Parameters and Run Button (this section is already defined above in the if statement)
        # For clarity, it means the st.subheader("LDA Parameters") and the "Run LDA Topic Modeling" button
        # and its checklist logic are part of this conditional block.
        # The critical part is that after a successful LDA run, the following is set:
        # if overall_success:
        #     st.session_state.show_detailed_dashboard_button = True
        #     st.session_state.lda_num_topics_run = num_topics_lda # Store the number of topics
        pass # The LDA run UI is already structured above.

# --- Initial state for dashboard view & Main conditional rendering logic ---
if 'viewing_dashboard' not in st.session_state: st.session_state.viewing_dashboard = False
if 'show_detailed_dashboard_button' not in st.session_state: st.session_state.show_detailed_dashboard_button = False
if "lda_num_topics_run" not in st.session_state: st.session_state.lda_num_topics_run = 10 

if st.session_state.get('viewing_dashboard'):
    if 'temp_output_dir_lda' in st.session_state and st.session_state.temp_output_dir_lda:
        render_detailed_dashboard(st.session_state.temp_output_dir_lda)
    else: 
        st.error("LDA results directory not found. Please run the LDA pipeline first.")
        if st.button("Go to Setup Page", key="dashboard_back_to_setup_main_btn"):
            st.session_state.viewing_dashboard = False; st.session_state.show_detailed_dashboard_button = False
            st.experimental_rerun()
else: 
    if st.session_state.get('show_detailed_dashboard_button'):
        if st.button("📊 View Detailed Analysis Dashboard", key="view_dashboard_button_main_page_main_btn"):
            st.session_state.viewing_dashboard = True; st.session_state.show_detailed_dashboard_button = False 
            st.experimental_rerun()
    elif df_original is None and not USER_TOPIC_MODELING_ERROR_MESSAGE: # If no file uploaded yet and no major import error
        st.info("☝️ Please upload a CSV file using the sidebar to begin.")
    elif df_original is not None and st.session_state.get('text_col_name') is None:
         st.warning("👈 Please select the 'Tweet/Text Column' in the sidebar to proceed with cleaning.")


# --- Footer and Manual Cleanup Button ---
if 'temp_output_dir_lda' in st.session_state and st.session_state.temp_output_dir_lda:
    st.sidebar.markdown("---")
    if st.sidebar.button("Manually Cleanup LDA Temp Directory", key="manual_cleanup_button_sidebar_final"):
        temp_dir_to_clean = st.session_state.temp_output_dir_lda
        if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
            try:
                shutil.rmtree(temp_dir_to_clean)
                st.sidebar.success(f"Successfully removed: {temp_dir_to_clean}")
                del st.session_state.temp_output_dir_lda
                st.session_state.viewing_dashboard = False 
                st.session_state.show_detailed_dashboard_button = False
                if 'lda_system_instance' in st.session_state: del st.session_state.lda_system_instance
                st.experimental_rerun()
            except Exception as e: st.sidebar.error(f"Error removing {temp_dir_to_clean}: {e}")
        else: st.sidebar.info("No active LDA temporary directory or path does not exist.")

st.markdown("---")
st.markdown("App by AI and Human with Topic Modeling Superpowers ✨")