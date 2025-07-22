import streamlit as st
import pandas as pd
import os
import utils
import modeling
import visualization
from sklearn.metrics.pairwise import cosine_similarity
import re

# --- Page Configuration ---
st.set_page_config(
    page_title="Multilingual Topic Modeling Dashboard",
    page_icon="🌍",
    layout="wide"
)

# --- Resource Setup ---
NLTK_READY = utils.ensure_nltk_resources()
if not NLTK_READY:
    st.error("Critical NLTK resources could not be loaded. Functionality will be limited. Please check the logs.")
    st.stop()

# --- Session State and Callback ---
if 'custom_stopwords_key' not in st.session_state: st.session_state.custom_stopwords_key = ""

def handle_stopwords_selection():
    all_selected_words = set()
    for key, value in st.session_state.items():
        if key.startswith("topic_multiselect_") and value:
            for word_with_weight in value:
                word_match = re.match(r"^(.*?)\s\(", word_with_weight)
                if word_match:
                    all_selected_words.add(word_match.group(1).strip())
    
    existing_stopwords = set(s.strip() for s in st.session_state.custom_stopwords_key.split(',') if s.strip())
    combined_stopwords = sorted(list(existing_stopwords.union(all_selected_words)))
    st.session_state.custom_stopwords_key = ", ".join(combined_stopwords)

# --- Main App ---
st.title("🌍 Multilingual Topic Modeling Dashboard")
st.markdown("Analyze textual data in **English, Spanish, Chinese, or Japanese** to discover topics and user trends.")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file:
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None

    df = pd.read_csv(uploaded_file, low_memory=False)
    with st.expander("1. Main Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Select Language & Columns")
            selected_language = st.selectbox("Select the language of your dataset", ["English", "Spanish", "Chinese", "Japanese"])
            column_options = ["<Select a Column>"] + df.columns.tolist()
            user_id_col = st.selectbox("Select User ID Column", options=column_options)
            content_col = st.selectbox("Select Text/Content Column *", options=column_options)
            datetime_col = st.selectbox("Select Date/Time Column", options=column_options)
        with col2:
            st.subheader("Select Model Parameters")
            num_topics = st.number_input("Select the number of topics", min_value=2, max_value=20, value=5)
            min_token_len = st.number_input("Min token length for analysis", min_value=1, max_value=20, value=2, help="Tokens shorter than this will be excluded.")
            
            st.markdown("---")
            use_phrases = st.checkbox("Detect & Combine Phrases (bigrams)", value=True, help="Automatically detect and combine common phrases like 'machine_learning'.")
            phrase_min_count = 0
            phrase_threshold = 0.0
            if use_phrases:
                phrase_min_count = st.slider("Phrase Min Count", min_value=1, max_value=100, value=10, help="Ignore words and bigrams with a total collected count lower than this value.")
                phrase_threshold = st.slider("Phrase Threshold", min_value=1.0, max_value=100.0, value=20.0, step=1.0, help="A higher threshold forms fewer phrases. See Gensim Phrases documentation for details.")

    with st.expander("2. Vocabulary & Feature Selection"):
        use_tfidf = st.checkbox("Use TF-IDF for Feature Selection", value=False, help="Filter vocabulary to keep only the most significant terms before LDA.")
        top_n_tfidf = 0
        if use_tfidf:
            top_n_tfidf = st.number_input("Keep Top N Words by TF-IDF", min_value=100, max_value=20000, value=2000, step=100)
        
        st.markdown("---")
        st.write("Vocabulary Filtering (if not using TF-IDF)")
        no_below = st.number_input("Min Document Frequency", min_value=1, max_value=100, value=5, help="Filter out tokens that appear in fewer than this many documents.", disabled=use_tfidf)
        no_above = st.slider("Max Document Frequency (%)", min_value=0.1, max_value=1.0, value=0.5, step=0.05, help="Filter out tokens that appear in more than this percentage of documents.", disabled=use_tfidf)

    with st.expander("3. Text Cleaning & Preprocessing Options"):
        # ... (This section is unchanged) ...
        cleaning_options = {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("---")
            cleaning_options['lowercase'] = st.checkbox("Convert to Lowercase", value=True)
            if selected_language == 'English':
                cleaning_options['lemmatize'] = st.checkbox("Lemmatize words (English only)", value=True)
            else:
                cleaning_options['lemmatize'] = False
            cleaning_options['remove_urls'] = st.checkbox("Remove URLs", value=True)
            cleaning_options['remove_html'] = st.checkbox("Remove HTML Tags", value=True)
            st.markdown("---")
            st.write("**Emoji Handling**")
            cleaning_options['emoji_handling'] = st.radio("Emoji Handling", ["Keep Emojis", "Remove Emojis", "Convert Emojis to Text"], index=1, label_visibility="collapsed")
        with c2:
            st.markdown("---")
            cleaning_options['remove_special_chars'] = st.checkbox("Remove Special Characters", value=True)
            cleaning_options['remove_punctuation'] = st.checkbox("Remove Punctuation", value=True)
            cleaning_options['remove_numbers'] = st.checkbox("Remove Numbers", value=True)
            cleaning_options['remove_stopwords'] = st.checkbox(f"Remove Stopwords ({selected_language})", value=True)
            st.markdown("---")
            st.write("**Hashtag (#) & Mention (@) Handling**")
            cleaning_options['hashtag_handling'] = st.radio("Hashtag Handling", ["Keep", "Remove", "Extract (new column)"], index=1, key="hashtag_handling", horizontal=True)
            cleaning_options['mention_handling'] = st.radio("Mention Handling", ["Keep", "Remove", "Extract (new column)"], index=1, key="mention_handling", horizontal=True)

        st.markdown("---")
        cleaning_options['custom_stopwords'] = st.text_area("Custom Stopwords (comma-separated)", key="custom_stopwords_key")
        cleaning_options['min_token_length'] = min_token_len


    if st.button("🚀 Run Topic Modeling Analysis", type="primary"):
        if any(col == "<Select a Column>" for col in [user_id_col, content_col, datetime_col]):
            st.warning("Please select all required columns in the Main Configuration section.")
        else:
            try:
                config = {
                    'user_col': user_id_col, 'content_col': content_col, 'date_col': datetime_col,
                    'num_topics': num_topics, 'language': selected_language,
                    'cleaning_options': cleaning_options,
                    'use_phrases': use_phrases,
                    'phrase_min_count': phrase_min_count,
                    'phrase_threshold': phrase_threshold,
                    'use_tfidf': use_tfidf,
                    'top_n_tfidf': top_n_tfidf,
                    'no_below': no_below,
                    'no_above': no_above
                }
                
                processed_df, analysis_results = modeling.run_analysis(df, config)
                
                if processed_df is not None and analysis_results is not None:
                    st.session_state.processed_data = processed_df
                    st.session_state.analysis_results = analysis_results
                    st.success("Analysis Complete!")
                    st.balloons()
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                st.exception(e)

if st.session_state.get('analysis_results'):
    processed_df = st.session_state.processed_data
    results = st.session_state.analysis_results
    config = results['config']

    st.divider()
    st.header("📈 Overall Project Summary")
    # ... (Summary metrics unchanged) ...
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

    with st.expander("🔎 View & Download Preprocessed Text"):
        st.markdown("This table shows the original text alongside the final list of tokens that were used for the topic model.")
        if 'tokenized_text' in processed_df.columns:
            display_df = processed_df[[config['content_col'], 'tokenized_text']].copy()
            display_df.rename(columns={config['content_col']: 'Original Text', 'tokenized_text': 'Processed Tokens'}, inplace=True)
            st.dataframe(display_df.head(100))
            csv_data = display_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="Download Preprocessed Text as CSV", data=csv_data, file_name='preprocessed_text.csv', mime='text/csv')

    st.header("Topic Model Evaluation")
    
    # NEW: Display all three evaluation metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        coherence_cv_val = results.get('coherence_score_cv')
        score_text, quality_text = visualization.interpret_coherence(coherence_cv_val)
        st.metric(label="Coherence (C_v)", value=score_text, help="Measures topic interpretability based on word co-occurrence. Higher is better.")
        st.caption(quality_text)
    
    with mcol2:
        coherence_umass_val = results.get('coherence_score_umass')
        score_text, quality_text = visualization.interpret_umass(coherence_umass_val)
        st.metric(label="Coherence (U_Mass)", value=score_text, help="Measures topic coherence based on document co-occurrence. Values are negative; closer to 0 is better.")
        st.caption(quality_text)

    with mcol3:
        perplexity_val = results.get('perplexity_score')
        score_text, quality_text = visualization.interpret_perplexity(perplexity_val)
        st.metric(label="Perplexity", value=score_text, help="Measures how well the model predicts a sample. It is influenced by vocabulary size.")
        st.caption(quality_text)
    
    with st.expander("📊 Topic Model Evaluation Metrics"):
        st.write("""
        ### 🔹Coherence Score
        - measures how well the discovered topics make sense:
        - **> 0.6**: Excellent - Topics are very distinct and meaningful
        - **0.5 - 0.6**: Good - Topics are generally clear and interpretable  
        - **0.4 - 0.5**: Fair - Topics are somewhat meaningful but may overlap
        - **< 0.4**: Poor - Topics may be unclear or too similar
        
        💡 **Tip**: If coherence is low, try adjusting the number of topics or cleaning options.
        ### 🔹 UMass Coherence Score
        - **Lower (more negative)** is better.
        - Typical range: **-0.2 to -2.0**.
        - Measures how often top words of a topic appear together in the same documents.
        - **More negative** values indicate **sharper, more coherent topics**.
        
        ---
        ### 🔸 Perplexity
        - A measure of how well the model predicts unseen data.
        - **Lower** perplexity is better.
        - However, perplexity **doesn't always align** with human interpretability.
        
        💡 **Tip:** A good model balances **high coherence** (human-friendly topics) with **low perplexity** (statistical accuracy).

        """)

    st.subheader("Topic Visualization")
    # ... (The rest of the display logic is unchanged) ...
    display_choice = st.radio("Choose how to view the topics:", ("Word Clouds", "Top 15 Words (Interactive Lists)"), horizontal=True, label_visibility="collapsed")
    
    font_path = None
    if config['language'] == 'Chinese': font_path = utils.CHINESE_FONT_PATH
    elif config['language'] == 'Japanese': font_path = utils.JAPANESE_FONT_PATH

    if display_choice == "Word Clouds":
        visualization.display_topic_wordclouds(results['lda_model'], results['feature_names'], config['num_topics'], font_path)
    else:
        st.info("Select words from the lists below to add them to the custom stopwords list above. Then, click 'Run Analysis' again.")
        cols = st.columns(min(config['num_topics'], 3))
        for topic_idx in range(config['num_topics']):
            with cols[topic_idx % 3]:
                st.subheader(f"Topic {topic_idx}")
                topic_weights = results['lda_model'].components_[topic_idx]
                top_features_ind = topic_weights.argsort()[:-15 - 1:-1]
                options_with_weights = [f"{results['feature_names'][i]} ({topic_weights[i]:.3f})" for i in top_features_ind]
                st.multiselect(
                    label=f"Top 15 words for Topic {topic_idx}",
                    options=options_with_weights,
                    key=f"topic_multiselect_{topic_idx}",
                    label_visibility="collapsed",
                    on_change=handle_stopwords_selection
                )
    
    st.divider()
    st.header("Topic Similarity Matrix")
    st.markdown("""
    This matrix shows the similarity between topics based on their word distributions. 
    - **Values closer to 1.0 (brighter colors)** indicate that two topics are very similar.
    - **Values closer to 0.0 (darker colors)** indicate that two topics are very distinct.
    
    💡 **Tip**: If you see high similarity between two topics (e.g., > 0.7), it might mean they are discussing similar themes. You could consider re-running the analysis with fewer topics.
    """)
    similarity_matrix = cosine_similarity(results['lda_model'].components_)
    visualization.display_similarity_matrix(similarity_matrix, config['num_topics'])
    
    st.divider()
    st.header("👤 User Topic Narrowness vs. Post Frequency")
    visualization.plot_user_scatter(results['user_analysis'], config['user_col'])

    st.divider()
    st.header("🔍 Deep Dive: Per-User Analysis")
    user_list = sorted(processed_df[config['user_col']].dropna().unique())
    selected_user = st.selectbox("Select a User to Analyze", options=user_list)
    
    if selected_user:
        user_data = processed_df[processed_df[config['user_col']] == selected_user]
        user_summary_stats = results['user_analysis'][results['user_analysis'][config['user_col']] == selected_user].iloc[0]
        st.subheader(f"Analysis for User: {selected_user}")

        user_full_distributions = results['full_topic_distributions']
        original_df_indices = df.index.tolist()
        user_indices_in_original_df = user_data.index.tolist()
        
        user_data = user_data.copy()
        user_data['topic_distribution'] = [user_full_distributions[original_df_indices.index(i)] for i in user_indices_in_original_df]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Posts by User", f"{int(user_summary_stats['post_count'])}")
            gini_val = user_summary_stats['gini_score']
            interpretation = ""
            if gini_val < 0.3: interpretation = "🌈 Diverse interests"
            elif gini_val < 0.7: interpretation = "🎯 Moderately focused"
            else: interpretation = "🔍 Highly specialized"
            st.metric("Topic Narrowness (Gini)", f"{gini_val:.3f}", help="Measures topic focus. 0 = balanced, 1 = specialized.")
            st.caption(interpretation)
            visualization.plot_user_donut(user_data)
        with col2:
            if len(user_data) > 1:
                visualization.plot_user_evolution_area(user_data, config['date_col'], config['num_topics'])
            else:
                st.info("A Topic Evolution chart requires more than one post.")
        st.subheader("User Posts")
        display_cols = [config['content_col'], config['date_col'], 'dominant_topic']
        if 'hashtags' in user_data.columns: display_cols.append('hashtags')
        if 'mentions' in user_data.columns: display_cols.append('mentions')
        st.dataframe(user_data[display_cols].rename(columns={config['content_col']: "Post Content", config['date_col']: "Timestamp", 'dominant_topic': 'Assigned Topic'}))
else:
    st.info("Please upload a file and run the analysis to see the results.")