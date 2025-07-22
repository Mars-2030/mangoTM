import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def interpret_coherence(score):
    """Provides a human-readable interpretation of a C_v coherence score."""
    if isinstance(score, str): return score, "Unable to calculate"
    if score > 0.6: return f"{score:.2f}", "🟢 Excellent"
    if score > 0.5: return f"{score:.2f}", "🟡 Good"
    if score > 0.4: return f"{score:.2f}", "🟠 Fair"
    return f"{score:.2f}", "🔴 Poor"

# NEW: Helper function for U_Mass coherence
def interpret_umass(score):
    """Provides a human-readable interpretation of a U_Mass coherence score."""
    if isinstance(score, str): return score, "Unable to calculate"
    # U_Mass is on a log scale, so values are negative. Closer to 0 is better.
    if score > -2: return f"{score:.2f}", "🟢 Excellent"
    if score > -5: return f"{score:.2f}", "🟡 Good"
    if score > -8: return f"{score:.2f}", "🟠 Fair"
    return f"{score:.2f}", "🔴 Poor"

# NEW: Helper function for Perplexity
def interpret_perplexity(score):
    """Provides a human-readable interpretation of a perplexity score."""
    if isinstance(score, str): return score, "Unable to calculate"
    return f"{score:,.0f}", "Lower is Better"

def display_topic_wordclouds(model, feature_names, num_topics, font_path=None):
    # ... (This function is unchanged) ...
    if font_path and not os.path.exists(font_path):
        st.warning(f"Font file not found at '{font_path}'. CJK characters may not display correctly.")
        font_path = None
    cols = st.columns(min(num_topics, 3))
    for topic_idx in range(num_topics):
        with cols[topic_idx % 3]:
            st.subheader(f"Topic {topic_idx}")
            topic_weights = model.components_[topic_idx]
            top_indices = topic_weights.argsort()[:-50 - 1:-1]
            word_freqs = {feature_names[i]: topic_weights[i] for i in top_indices if i < len(feature_names)}
            if not word_freqs:
                st.write("No significant words found for this topic.")
                continue
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(word_freqs)
            fig, ax = plt.subplots(figsize=(10, 5)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig); plt.close(fig)

def display_similarity_matrix(similarity_matrix, num_topics):
    # ... (This function is unchanged) ...
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(
        similarity_matrix, annot=True, fmt=".2f", cmap='viridis', 
        xticklabels=[f"Topic {i}" for i in range(num_topics)],
        yticklabels=[f"Topic {i}" for i in range(num_topics)],
        annot_kws={"size": 8}
    )
    ax.set_title("Topic Similarity Matrix", fontsize=12)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)

def plot_user_scatter(user_analysis_df, user_col):
    # ... (This function is unchanged) ...
    fig = px.scatter(
        user_analysis_df, x='post_count', y='gini_score', hover_data=[user_col], 
        labels={'post_count': 'Number of Posts', 'gini_score': 'Gini Coefficient (Topic Narrowness)'}, 
        title="User Engagement Profile"
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_user_donut(user_data):
    # ... (This function is unchanged) ...
    if user_data.empty or 'dominant_topic' not in user_data.columns:
        st.info("No topic data available for this user.")
        return
    topic_counts = user_data['dominant_topic'].dropna().value_counts().sort_index()
    if topic_counts.empty:
        st.info("No topics assigned to this user's posts.")
        return
    fig = go.Figure(data=[go.Pie(
        labels=[f"Topic {int(i)}" for i in topic_counts.index], values=topic_counts.values, 
        hole=.4, hoverinfo='label+percent', textinfo='percent'
    )])
    fig.update_layout(title_text="Topic Distribution for User", showlegend=True, legend_title_text="Topics")
    st.plotly_chart(fig, use_container_width=True)

def plot_user_evolution_area(user_data, date_col, num_topics):
    # ... (This function is unchanged) ...
    evolution_df = user_data.sort_values(by=date_col).copy()
    if 'topic_distribution' not in evolution_df.columns or evolution_df['topic_distribution'].isnull().all():
        st.info("Not enough data to plot topic evolution.")
        return
    topic_props = pd.DataFrame(evolution_df['topic_distribution'].to_list(), columns=[f"Topic_{i}" for i in range(num_topics)], index=evolution_df.index)
    evolution_df = pd.concat([evolution_df[[date_col]], topic_props], axis=1)
    fig = px.area(
        evolution_df, x=date_col, y=topic_props.columns, 
        title=f"Topic Evolution Over Time for User (Stacked)", 
        labels={'value': 'Topic Proportion', date_col: 'Time', 'variable': 'Topic'}
    )
    fig.update_yaxes(range=[0, 1]); st.plotly_chart(fig, use_container_width=True)