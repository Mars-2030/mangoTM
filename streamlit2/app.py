import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import jieba
import emoji # New import for demojizing

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual Topic Modeling Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- Constants & NLTK Setup ---
CHINESE_FONT_PATH = 'streamlit2/NotoSansSC-Regular.ttf'
CHINESE_STOPWORDS_PATH = 'cn_stopwords.txt'

@st.cache_resource
def download_nltk_data():
    for resource in ['stopwords', 'punkt']:
        try: nltk.data.find(f'corpora/{resource}')
        except LookupError: nltk.download(resource)
download_nltk_data()

@st.cache_data
def load_chinese_stopwords():
    if not os.path.exists(CHINESE_STOPWORDS_PATH): return set()
    with open(CHINESE_STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        return {line.strip() for line in f}

# --- Text Processing and Helper Functions ---

# NEW: Functions for extraction features
def extract_hashtags(text):
    return re.findall(r"#(\w+)", text)

def extract_mentions(text):
    return re.findall(r"@(\w+)", text)

# REWRITTEN: A comprehensive function that handles all new cleaning options
def clean_and_tokenize(text, lang, options):
    if pd.isna(text): return []
    text = str(text)

    # 1. Extraction (is done on the DataFrame before this function is called)
    # 2. Early cleaning
    if options['lowercase']: text = text.lower()
    if options['remove_html']: text = re.sub(r'<.*?>', '', text)
    if options['remove_urls']: text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Handling Mentions and Hashtags (based on user choice)
    if options['mention_handling'] == 'Remove Mentions':
        text = re.sub(r'@\w+', '', text)
    if options['hashtag_handling'] == 'Remove Hashtags':
        text = re.sub(r'#\w+', '', text)

    # 4. Emoji Handling
    if options['emoji_handling'] == 'Remove Emojis':
        text = emoji.replace_emoji(text, replace='')
    elif options['emoji_handling'] == 'Convert Emojis to Text':
        text = emoji.demojize(text)

    # 5. Final Character Cleaning
    if options['remove_special_chars']: text = re.sub(r'[^a-zA-Z0-9\s_]', '', text) # Keep underscore for demojized emojis
    if options['remove_punctuation']: text = re.sub(r'[^\w\s]', '', text)
    if options['remove_numbers']: text = re.sub(r'\d+', '', text)
    
    # 6. Tokenization (language-specific)
    if lang == 'Chinese':
        tokens = jieba.lcut(text)
    else:
        tokens = word_tokenize(text)

    # 7. Stopword Removal (standard + custom)
    if options['remove_stopwords']:
        stop_words = set()
        if lang == 'English': stop_words.update(stopwords.words('english'))
        elif lang == 'Spanish': stop_words.update(stopwords.words('spanish'))
        elif lang == 'Chinese': stop_words.update(load_chinese_stopwords())
        
        if options.get('custom_stopwords'):
            custom_list = [word.strip() for word in options['custom_stopwords'].split(',') if word.strip()]
            stop_words.update(custom_list)
        
        tokens = [token for token in tokens if token not in stop_words]

    # 8. Filter by minimum token length
    min_len = options.get('min_token_length', 1)
    tokens = [token for token in tokens if len(token) >= min_len]

    return tokens

# Other helper functions remain the same
def gini(array):
    if np.sum(array) == 0: return 0.0
    array = np.array(array, dtype=np.float64); array = np.sort(array)
    index = np.arange(1,array.shape[0]+1); n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def display_topic_wordclouds(model, feature_names, num_topics, font_path=None):
    if font_path and not os.path.exists(font_path):
        st.warning(f"Font file not found at '{font_path}'. Chinese characters may not display correctly.")
        font_path = None
    cols = st.columns(min(num_topics, 3))
    for topic_idx in range(num_topics):
        with cols[topic_idx % 3]:
            st.subheader(f"Topic {topic_idx}")
            topic_weights = model.components_[topic_idx]
            word_freqs = {feature_names[i]: topic_weights[i] for i in topic_weights.argsort()[:-50 - 1:-1]}
            if not word_freqs:
                st.write("No significant words found for this topic.")
                continue
            wc = WordCloud(width=800, height=400, background_color='white', font_path=font_path).generate_from_frequencies(word_freqs)
            fig, ax = plt.subplots(figsize=(10,5)); ax.imshow(wc, interpolation='bilinear'); ax.axis('off'); st.pyplot(fig); plt.close(fig)

def interpret_coherence(score):
    if isinstance(score, str): return score, "Unable to calculate"
    if score > 0.6: return f"{score:.2f}", "🟢 Excellent"
    if score > 0.5: return f"{score:.2f}", "🟡 Good"
    if score > 0.4: return f"{score:.2f}", "🟠 Fair"
    return f"{score:.2f}", "🔴 Poor"

# --- Session State Initialization ---
if 'processed_data' not in st.session_state: st.session_state.processed_data = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

# --- Main Page Display ---
st.title("🌍 Multilingual Topic Modeling Dashboard")
st.markdown("Analyze textual data in **English, Spanish, or Chinese** to discover topics and user trends.")

# --- Configuration Section ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file, low_memory=False)
    with st.expander("1. Main Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select Language & Columns")
            selected_language = st.selectbox("Select the language of your dataset", ["English", "Spanish", "Chinese"])
            column_options = ["<Select a Column>"] + df.columns.tolist()
            user_id_col = st.selectbox("Select User ID Column", options=column_options)
            content_col = st.selectbox("Select Text/Content Column *", options=column_options)
            datetime_col = st.selectbox("Select Date/Time Column", options=column_options)
        with col2:
            st.subheader("Select Model Parameters")
            num_topics = st.number_input("Select the number of topics", min_value=2, max_value=20, value=5)
                                            #  help="Tokens shorter than this will be excluded from the analysis.")
            min_token_len = st.number_input("Min token length for LDA document", min_value=1, max_value=20, value=5, 
                                             help="Tokens shorter than this will be excluded from the analysis.")

    # --- NEW: Overhauled Cleaning Options Expander ---
    with st.expander("2. Text Cleaning & Preprocessing Options"):
        st.markdown(f"Cleaning options for column: **{content_col}**")
        cleaning_options = {}
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("---")
            cleaning_options['lowercase'] = st.checkbox("Convert to Lowercase", value=True)
            cleaning_options['remove_urls'] = st.checkbox("Remove URLs", value=True)
            cleaning_options['remove_html'] = st.checkbox("Remove HTML Tags", value=True)
            
            st.markdown("---")
            st.write("**Emoji Handling**")
            cleaning_options['emoji_handling'] = st.radio(
                "Emoji Handling",
                ["Keep Emojis", "Remove Emojis", "Convert Emojis to Text"],
                index=1, label_visibility="collapsed"
            )
            st.markdown("---")
            st.write("**Hashtag (#) Handling**")
            cleaning_options['hashtag_handling'] = st.radio(
                "Hashtag Handling",
                ["Keep Hashtags", "Remove Hashtags", "Extract Hashtags (new column)"],
                index=1, label_visibility="collapsed"
            )
            st.markdown("---")
            st.write("**Mention (@) Handling**")
            cleaning_options['mention_handling'] = st.radio(
                "Mention Handling",
                ["Keep Mentions", "Remove Mentions", "Extract Mentions (new column)"],
                index=1, label_visibility="collapsed"
            )

        with c2:
            st.markdown("---")
            cleaning_options['remove_special_chars'] = st.checkbox("Remove Special Characters - removes symbols like !@#$%^&*()", value=True)
            cleaning_options['remove_punctuation'] = st.checkbox("Remove Punctuation (is more aggressive and removes all non-alphanumeric)", value=True)
            cleaning_options['remove_numbers'] = st.checkbox("Remove Numbers", value=True)
            st.markdown("---")
            
            cleaning_options['remove_stopwords'] = st.checkbox(f"Remove Stopwords ({selected_language})", value=True)
            cleaning_options['custom_stopwords'] = st.text_area(
                "Custom Stopwords (comma-separated)",
                help="Enter additional stopwords to remove, separated by commas. e.g., apple,microsoft,google"
            )
            
        cleaning_options['min_token_length'] = min_token_len

    if st.button("🚀 Run Topic Modeling Analysis", type="primary"):
        if any(col == "<Select a Column>" for col in [user_id_col, content_col, datetime_col]):
            st.warning("Please select all required columns in the Main Configuration section.")
        else:
            try:
                progress_bar = st.progress(0, text="Step 1/5: Validating and cleaning data...")
                analysis_df = df.copy()
                analysis_df.dropna(subset=[user_id_col, content_col, datetime_col], inplace=True)
                analysis_df[datetime_col] = pd.to_datetime(analysis_df[datetime_col], errors='coerce')
                analysis_df.dropna(subset=[datetime_col], inplace=True)
                
                # --- NEW: Pre-processing for extraction ---
                if cleaning_options['hashtag_handling'] == 'Extract Hashtags (new column)':
                    analysis_df['hashtags'] = analysis_df[content_col].astype(str).apply(extract_hashtags)
                if cleaning_options['mention_handling'] == 'Extract Mentions (new column)':
                    analysis_df['mentions'] = analysis_df[content_col].astype(str).apply(extract_mentions)

                if analysis_df.empty:
                    st.error("Error: No valid data remains after initial cleaning.")
                    st.stop()
                
                progress_bar.progress(20, text="Step 2/5: Tokenizing text with advanced options...")
                tokenized_docs = analysis_df[content_col].apply(
                    lambda text: clean_and_tokenize(text, selected_language, cleaning_options)
                ).tolist()
                
                progress_bar.progress(40, text="Step 3/5: Vectorizing text with TF-IDF...")
                tokenized_docs_str = [' '.join(doc) for doc in tokenized_docs]
                try:
                    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, lowercase=False)
                    tfidf_matrix = vectorizer.fit_transform(tokenized_docs_str)
                    if tfidf_matrix.shape[1] == 0:
                        st.error("Error: The vocabulary is empty after text cleaning. Please adjust the cleaning options.")
                        st.stop()
                    feature_names = vectorizer.get_feature_names_out()
                except ValueError:
                    st.error("Error: Could not create a document-term matrix. This often happens if all documents are empty after cleaning.")
                    st.stop()

                progress_bar.progress(60, text="Step 4/5: Fitting LDA model...")
                lda = LatentDirichletAllocation(n_components=num_topics, random_state=42, n_jobs=1)
                lda.fit(tfidf_matrix)
                topic_results = lda.transform(tfidf_matrix)
                analysis_df['dominant_topic'] = np.argmax(topic_results, axis=1)
                analysis_df['topic_distribution'] = list(topic_results)

                progress_bar.progress(80, text="Step 5/5: Calculating coherence and finalizing...")
                try:
                    dictionary = Dictionary(tokenized_docs)
                    topics = [[feature_names[i] for i in topic.argsort()[:-20 - 1:-1]] for topic in lda.components_]
                    coherence_model = CoherenceModel(topics=topics, texts=tokenized_docs, dictionary=dictionary, coherence='c_v')
                    coherence_score = coherence_model.get_coherence()
                except Exception as e:
                    st.warning(f"Could not calculate coherence score: {e}")
                    coherence_score = "N/A"
                
                user_summary = analysis_df.groupby(user_id_col)['topic_distribution'].apply(lambda dists: np.mean(dists, axis=0))
                user_analysis = pd.DataFrame({'post_count': analysis_df.groupby(user_id_col).size(), 'gini_score': user_summary.apply(gini)}).reset_index()
                
                st.session_state.processed_data = analysis_df
                st.session_state.analysis_results = {
                    'lda_model': lda, 'feature_names': feature_names, 'user_analysis': user_analysis, 'coherence_score': coherence_score,
                    'config': {'user_col': user_id_col, 'date_col': datetime_col, 'content_col': content_col, 'num_topics': num_topics, 'language': selected_language}
                }
                progress_bar.progress(100, text="Analysis Complete!")
                st.success("Analysis Complete!")
                st.balloons()
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.exception(e)

# --- Display Results Section ---
if st.session_state.analysis_results:
    processed_df = st.session_state.processed_data
    results = st.session_state.analysis_results
    config = results['config']

    # (The results display section remains largely the same, but will now show the new extracted columns in the final dataframe)
    st.divider()
    st.header("📈 Overall Project Summary")
    num_users = processed_df[config['user_col']].nunique()
    total_posts = len(processed_df)
    avg_posts_per_user = total_posts / num_users if num_users > 0 else 0
    date_range = f"{processed_df[config['date_col']].min().strftime('%Y/%m/%d')} - {processed_df[config['date_col']].max().strftime('%Y/%m/%d')}"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("# Users", f"{num_users}")
    col2.metric("Total Posts", f"{total_posts}")
    col3.metric("Avg Posts/User", f"{avg_posts_per_user:.1f}")
    col4.metric("Date Range", date_range)
    
    st.divider()
    st.header("Topic Modeling Analysis")
    coherence_val = results.get('coherence_score')
    score_text, quality_text = interpret_coherence(coherence_val)
    st.markdown(f"The content in your data has been broken into **{config['num_topics']} topics** with a **Coherence Score: {score_text}** ({quality_text})")
    
    if isinstance(coherence_val, float):
        if coherence_val < 0.4:
            st.warning("⚠️ Low coherence detected. Consider reducing the number of topics or adjusting text cleaning options for better results.")
        elif coherence_val > 0.65:
            st.success("✅ High coherence! Your topics are well-defined and meaningful.")
    with st.expander("📊 Understanding Coherence Score"):
     st.write("""
    **Coherence Score** measures how well the discovered topics make sense:
    
    - **> 0.6**: Excellent - Topics are very distinct and meaningful
    - **0.5 - 0.6**: Good - Topics are generally clear and interpretable  
    - **0.4 - 0.5**: Fair - Topics are somewhat meaningful but may overlap
    - **< 0.4**: Poor - Topics may be unclear or too similar
    
    💡 **Tip**: If coherence is low, try adjusting the number of topics or cleaning options.
    """)
    
    font_path = CHINESE_FONT_PATH if config['language'] == 'Chinese' else None
    display_topic_wordclouds(results['lda_model'], results['feature_names'], config['num_topics'], font_path)

    st.divider()
    st.header("👤 User Topic Narrowness vs. Post Frequency")
    fig_scatter = px.scatter(
        results['user_analysis'], x='post_count', y='gini_score',
        hover_data=[config['user_col']],
        labels={'post_count': 'Number of Posts', 'gini_score': 'Gini Coefficient (Topic Narrowness)'},
        title="User Engagement Profile"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.header("🔍 Deep Dive: Per-User Analysis")
    user_list = sorted(processed_df[config['user_col']].unique())
    selected_user = st.selectbox("Select a User to Analyze", options=user_list)
    
    if selected_user:
        user_data = processed_df[processed_df[config['user_col']] == selected_user].copy()
        user_summary_stats = results['user_analysis'][results['user_analysis'][config['user_col']] == selected_user].iloc[0]

        st.subheader(f"Analysis for User: {selected_user}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Posts by User", f"{int(user_summary_stats['post_count'])}")
            gini_val = user_summary_stats['gini_score']
            if gini_val < 0.3: interpretation = "🌈 Diverse interests"
            elif gini_val < 0.7: interpretation = "🎯 Moderately focused"
            else: interpretation = "🔍 Highly specialized"
            st.metric("Topic Narrowness (Gini)", f"{gini_val:.3f}", 
                      help="Measures how focused a user's posts are. 0 = perfectly balanced, 1 = focused on one topic.")
            st.caption(interpretation)
            topic_counts = user_data['dominant_topic'].value_counts().sort_index()
            fig_donut = go.Figure(data=[go.Pie(
                labels=[f"Topic {i}" for i in topic_counts.index],
                values=topic_counts.values, hole=.4, hoverinfo='label+percent', textinfo='percent'
            )])
            fig_donut.update_layout(title_text="Topic Distribution for User", showlegend=True, legend_title_text="Topics")
            st.plotly_chart(fig_donut, use_container_width=True)
        with col2:
            user_data['date'] = user_data[config['date_col']].dt.date
            if user_data['date'].nunique() > 1:
                evolution_data = user_data.groupby(['date', 'dominant_topic']).size().unstack(fill_value=0)
                for i in range(config['num_topics']):
                    if i not in evolution_data.columns: evolution_data[i] = 0
                evolution_data = evolution_data.sort_index(axis=1)
                evolution_data.columns = [f"Topic {c}" for c in evolution_data.columns]
                fig_area = px.area(
                    evolution_data, x=evolution_data.index, y=evolution_data.columns,
                    title="User's Topic Evolution Over Time",
                    labels={'value': 'Number of Posts', 'date': 'Date', 'variable': 'Topic'}
                )
                st.plotly_chart(fig_area, use_container_width=True)
            else:
                st.info("A Topic Evolution chart cannot be generated as this user has only posted on a single day.")
        
        st.subheader("User Posts")
        # Show the new extracted columns if they exist
        display_cols = [config['content_col'], config['date_col'], 'dominant_topic']
        if 'hashtags' in user_data.columns: display_cols.append('hashtags')
        if 'mentions' in user_data.columns: display_cols.append('mentions')
        st.dataframe(user_data[display_cols].rename(columns={
            config['content_col']: "Post Content", config['date_col']: "Timestamp", 'dominant_topic': 'Assigned Topic'
        }))
else:
    st.info("Please expand the '⚙️ Configuration & Setup' section above, upload a file, and run the analysis to see the results.")