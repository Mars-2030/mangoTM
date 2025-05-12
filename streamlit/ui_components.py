# ui_components.py
import streamlit as st
import pandas as pd
import re
from bs4 import BeautifulSoup
import emoji
import string
# nltk.corpus.stopwords will be passed as an argument where needed

# --- Text Preprocessing Functions ---
def remove_urls(text: str) -> str:
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

def remove_html(text: str) -> str:
    return BeautifulSoup(text, "html.parser").get_text()

def remove_special_chars(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9\s.,!?:;\'-@#_]', '', text)

def remove_hashtags_from_text(text: str) -> str:
    return re.sub(r'#\w+', '', text)

def extract_hashtags(text: str) -> list:
    return re.findall(r'#\w+', text)

def remove_mentions_from_text(text: str) -> str:
    return re.sub(r'@\w+', '', text)

def extract_mentions(text: str) -> list:
    return re.findall(r'@\w+', text)

def remove_emojis_func(text: str) -> str: # Renamed to avoid conflict
    return emoji.replace_emoji(text, replace='')

def demojize_emojis_func(text: str) -> str: # Renamed
    return emoji.demojize(text)

def convert_to_lowercase(text: str) -> str:
    return text.lower()

def remove_stopwords_nltk_ui(text: str, custom_stopwords: list = None, nltk_stopwords_module=None) -> str:
    if not nltk_stopwords_module:
        # This function is now more of a utility; actual warning should be handled by caller if needed
        return str(text) 
    try:
        stop_words_set = set(nltk_stopwords_module.words('english'))
        if custom_stopwords:
            custom_stopwords_lower = [stop.lower() for stop in custom_stopwords]
            stop_words_set.update(custom_stopwords_lower)
        
        words = str(text).split() 
        filtered_words = [word for word in words if word.lower() not in stop_words_set]
        return " ".join(filtered_words)
    except Exception: # Broad exception if nltk_stopwords_module is not as expected
        return str(text)

def remove_numbers_from_text(text: str) -> str:
    return re.sub(r'\d+', '', text)

def remove_punctuation_from_text(text: str) -> str:
    return str(text).translate(str.maketrans('', '', string.punctuation.replace('@','').replace('#','')))

# --- UI Rendering Functions ---

def display_nltk_messages_from_session():
    """Displays NLTK resource messages stored in session_state within the sidebar."""
    if 'nltk_messages_for_sidebar' in st.session_state and st.session_state.nltk_messages_for_sidebar:
        expanded_nltk_status = not st.session_state.get('nltk_resources_all_good', True) or \
                               any("INFO: Downloading" in msg for msg in st.session_state.nltk_messages_for_sidebar)
        
        with st.sidebar.expander("NLTK Resource Status", expanded=expanded_nltk_status): 
            for msg in st.session_state.nltk_messages_for_sidebar:
                if msg.startswith("ERROR:") : st.error(msg)
                elif msg.startswith("INFO:") : st.info(msg)
                else: st.success(msg)
        
        # Clear messages after displaying to prevent re-display on every interaction
        # This ensures they only show up once per "ensure_nltk_resources" run effectively
        st.session_state.nltk_messages_for_sidebar = []


def render_sidebar_setup(df_loaded_in_main_app): # Takes the loaded DataFrame as argument
    """Renders the column selection widgets in the sidebar.
       Assumes df_loaded_in_main_app is the DataFrame loaded from the file uploader in main_app.py.
       Returns: user_id_col_name, text_col_name, datetime_col_name
    """
    user_id_col_name, text_col_name, datetime_col_name = None, None, None

    if df_loaded_in_main_app is not None:
        column_options = ["<Select a Column>"] + df_loaded_in_main_app.columns.tolist()
        
        def find_default_index(cols, common_names, default_val=0):
            for name in common_names:
                if name in cols:
                    return cols.index(name)
            return default_val

        user_id_col = st.sidebar.selectbox(
            "Select User ID Column (Needed for Topic Modeling)", 
            column_options, 
            index=find_default_index(column_options, ['user_id', 'userid', 'author', 'USER_ID']), # Added more common names
            key="ui_user_id_sel_comp_v2"
        )
        text_col = st.sidebar.selectbox(
            "Select Text/Content Column *", 
            column_options, 
            index=find_default_index(column_options, ['post_content', 'text', 'tweet', 'content', 'body', 'TEXT', 'TWEET']), # Added more common names
            key="ui_text_col_sel_comp_v2"
        )
        datetime_col = st.sidebar.selectbox(
            "Select Date/Time Column (Needed for Topic Modeling)", 
            column_options, 
            index=find_default_index(column_options, ['timestamp', 'date', 'created_at', 'time', 'TIMESTAMP']), # Added more common names
            key="ui_datetime_col_sel_comp_v2"
        )
        
        user_id_col_name = user_id_col if user_id_col != "<Select a Column>" else None
        text_col_name = text_col if text_col != "<Select a Column>" else None
        datetime_col_name = datetime_col if datetime_col != "<Select a Column>" else None

        if not text_col_name:
            st.sidebar.warning("Please select the Text/Content column to enable further processing.")
        
        # Store in session_state for global access if needed by other parts of main_app.py
        st.session_state.user_id_col_name = user_id_col_name
        st.session_state.text_col_name = text_col_name
        st.session_state.datetime_col_name = datetime_col_name
    
    return user_id_col_name, text_col_name, datetime_col_name

def render_cleaning_options_sidebar():
    """Renders cleaning option widgets in the sidebar.
       Returns a dictionary of selected cleaning options.
    """
    # Header will be called in main_app.py
    # st.sidebar.header("2. Cleaning Options")
    if not st.session_state.get('text_col_name'):
        st.sidebar.info("Select a text column first to see cleaning options.")
        return {} # Return empty if no text column selected

    st.sidebar.write(f"Cleaning options for column: **{st.session_state.get('text_col_name')}**")
    
    options = {}
    options['lowercase'] = st.sidebar.checkbox("Convert to Lowercase", True, key="ui_clean_lowercase_cb")
    options['remove_urls'] = st.sidebar.checkbox("Remove URLs", True, key="ui_clean_urls_cb")
    options['remove_html'] = st.sidebar.checkbox("Remove HTML Tags", True, key="ui_clean_html_cb")
    options['emoji_handling'] = st.sidebar.radio("Emoji Handling", ("Keep Emojis", "Remove Emojis", "Convert Emojis to Text (Demojize)"), index=1, key="ui_clean_emoji_radio")
    
    st.sidebar.markdown("---")
    options['hashtag_option'] = st.sidebar.radio("Hashtag (#) Handling", ("Keep Hashtags", "Remove Hashtags (symbol & text)", "Extract Hashtags (new column)"), index=1, key="ui_clean_hashtag_radio")
    options['mention_option'] = st.sidebar.radio("Mention (@) Handling", ("Keep Mentions", "Remove Mentions (symbol & text)", "Extract Mentions (new column)"), index=1, key="ui_clean_mention_radio")
    
    st.sidebar.markdown("---")
    options['remove_special_chars'] = st.sidebar.checkbox("Remove Special Characters", False, key="ui_clean_special_cb")
    options['remove_punctuation'] = st.sidebar.checkbox("Remove Punctuation (aggressive)", False, key="ui_clean_punct_cb")
    options['remove_numbers'] = st.sidebar.checkbox("Remove Numbers", False, key="ui_clean_numbers_cb")
    options['remove_stopwords'] = st.sidebar.checkbox("Remove Stopwords (English)", False, key="ui_clean_stopwords_cb")
    options['custom_stopwords_str'] = st.sidebar.text_area("Custom Stopwords (comma-separated)", help="e.g., app,rt,via", key="ui_custom_stops_text_area")
    
    return options

def perform_text_cleaning(df, text_column_name, cleaning_opts, nltk_stopwords_module_passed=None):
    """Performs text cleaning on the specified DataFrame column based on options."""
    if df is None or text_column_name not in df.columns:
        st.error("DataFrame or text column not provided for cleaning.")
        return None

    processed_df = df.copy()
    # Ensure the text column is treated as string, handle potential NaNs by converting to empty string
    text_series = processed_df[text_column_name].fillna('').astype(str)


    if cleaning_opts.get('lowercase'): text_series = text_series.apply(convert_to_lowercase)
    if cleaning_opts.get('remove_html'): text_series = text_series.apply(remove_html)
    if cleaning_opts.get('remove_urls'): text_series = text_series.apply(remove_urls)

    emoji_opt = cleaning_opts.get('emoji_handling')
    if emoji_opt == "Remove Emojis": text_series = text_series.apply(remove_emojis_func)
    elif emoji_opt == "Convert Emojis to Text (Demojize)": text_series = text_series.apply(demojize_emojis_func)

    hashtag_opt = cleaning_opts.get('hashtag_option')
    if hashtag_opt == "Extract Hashtags (new column)":
        processed_df['extracted_hashtags'] = text_series.apply(extract_hashtags)
    elif hashtag_opt == "Remove Hashtags (symbol & text)":
        text_series = text_series.apply(remove_hashtags_from_text) # Use renamed function

    mention_opt = cleaning_opts.get('mention_option')
    if mention_opt == "Extract Mentions (new column)":
        processed_df['extracted_mentions'] = text_series.apply(extract_mentions)
    elif mention_opt == "Remove Mentions (symbol & text)":
        text_series = text_series.apply(remove_mentions_from_text) # Use renamed function
    
    # Ensure these are applied in a sensible order (e.g. special chars before punctuation if needed)
    if cleaning_opts.get('remove_special_chars'): text_series = text_series.apply(remove_special_chars)
    if cleaning_opts.get('remove_punctuation'): text_series = text_series.apply(remove_punctuation_from_text) # Use renamed

    if cleaning_opts.get('remove_numbers'): text_series = text_series.apply(remove_numbers_from_text) # Use renamed

    if cleaning_opts.get('remove_stopwords'):
        custom_stops_list = [s.strip().lower() for s in cleaning_opts.get('custom_stopwords_str', '').split(',') if s.strip()]
        text_series = text_series.apply(
            lambda x: remove_stopwords_nltk_ui(
                x, 
                custom_stopwords=custom_stops_list,
                nltk_stopwords_module=nltk_stopwords_module_passed
            )
        )
    
    cleaned_col_name = f"cleaned_{text_column_name}"
    processed_df[cleaned_col_name] = text_series
    st.session_state.cleaned_text_col_name = cleaned_col_name # Store for lda_orchestrator

    return processed_df


def render_lda_parameters_sidebar():
    """Renders LDA parameter input widgets in the sidebar.
       Returns a dictionary of LDA parameters.
    """
    # Header called in main_app.py
    # st.sidebar.header("3. LDA Topic Modeling")
    params = {}
    params['num_topics'] = st.sidebar.number_input("Number of Topics (LDA)", min_value=2, max_value=100, value=st.session_state.get("lda_num_topics_run", 10), step=1, key="ui_lda_num_topics_input")
    params['time_bin'] = st.sidebar.selectbox("Time Bin for Temporal Analysis", ['day', 'week', 'month'], index=1, key="ui_lda_time_bin_select")
    params['combine_posts'] = st.sidebar.checkbox("Combine Posts by User & Time Bin", True, key="ui_lda_combine_posts_cb")
    params['min_post_length'] = st.sidebar.number_input("Min token length for LDA document", min_value=1, value=3, step=1, key="ui_lda_min_post_len_input")
    default_lda_stopwords = "rt,amp,via,http,https,co,u,wa,ha,thi,doe,becau,veri,ani,becaus,abov,ani,alon,alreadi,alway,anoth,anyon,anyth,anywher,aras,area,ask,avail,awai,back,becam,becom,befor,began,begin,begun,believ,besid,better,bigger,biggest,blockquot,nbsp"
    params['extra_stopwords_lda'] = st.sidebar.text_area("Additional Stopwords for LDA model (comma-separated)", default_lda_stopwords, key="ui_lda_extra_stopwords_area")
    params['run_visualizations'] = st.sidebar.checkbox("Generate & Save Visualizations by LDA system", True, key="ui_lda_run_viz_cb") 
    params['run_suspicious_detection'] = st.sidebar.checkbox("Detect Suspicious Users by LDA system", True, key="ui_lda_run_suspicious_cb") 
    
    return params