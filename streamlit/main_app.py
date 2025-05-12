# main_app.py
import streamlit as st

# 1. Call st.set_page_config() FIRST
st.set_page_config(layout="wide", page_title="Social Media Text Processor & Analyzer")

# 2. Then other imports
import pandas as pd
import os
import traceback
import nltk 

# --- Attempt to import custom modules ---
MODULES_AVAILABLE = False
UI_COMPONENTS_LOADED = False
LDA_ORCHESTRATOR_LOADED = False
DASHBOARD_RENDERERS_LOADED = False
USER_TOPIC_MODELING_CLASS_LOADED = False

try:
    from ui_components import (
        render_sidebar_setup, 
        display_nltk_messages_from_session,
        render_cleaning_options_sidebar, 
        perform_text_cleaning, 
        render_lda_parameters_sidebar
    )
    UI_COMPONENTS_LOADED = True
    from lda_pipeline_orchestrator import run_lda_pipeline_with_checklist # This function needs UserTopicModelingSystem
    LDA_ORCHESTRATOR_LOADED = True
    from dashboard_renderers import render_full_dashboard # This needs plotting functions
    DASHBOARD_RENDERERS_LOADED = True
    MODULES_AVAILABLE = UI_COMPONENTS_LOADED and LDA_ORCHESTRATOR_LOADED and DASHBOARD_RENDERERS_LOADED
except ImportError as e_mod:
    st.error(f"Failed to import one or more application modules (ui_components, lda_pipeline_orchestrator, dashboard_renderers). "
             f"Ensure these .py files exist in the same directory ('{os.getcwd()}') and have no internal import errors. Error: {e_mod}")
    MODULES_AVAILABLE = False

# --- NLTK Resource Check and Download Function ---

@st.cache_resource
@st.cache_resource
def ensure_nltk_resources():
    import shutil
    import tempfile
    from pathlib import Path

    try:
        app_nltk_data_path = Path(os.getcwd()) / "nltk_data_streamlit"
        app_nltk_data_path.mkdir(parents=True, exist_ok=True)
    except Exception as e_mkdir:
        app_nltk_data_path = Path(tempfile.gettempdir()) / "streamlit_nltk_data"
        app_nltk_data_path.mkdir(parents=True, exist_ok=True)
        st.warning(f"Could not create nltk_data_streamlit in app root. Using temp dir: {app_nltk_data_path}. Error: {e_mkdir}")

    if str(app_nltk_data_path) not in nltk.data.path:
        nltk.data.path.insert(0, str(app_nltk_data_path))

    resources_to_check = {
        "stopwords": ("corpora/stopwords", "stopwords"),
        "punkt": ("tokenizers/punkt", "punkt"),
        "wordnet": ("corpora/wordnet", "wordnet"),
    }

    all_good = True
    messages = []

    messages.append(f"INFO: NLTK will search for data in these paths (priority order): {nltk.data.path}")
    messages.append(f"INFO: NLTK downloads (if needed) will target: {app_nltk_data_path}")

    for name, (path_suffix, package_id) in resources_to_check.items():
        try:
            nltk.data.find(path_suffix)
            messages.append(f"SUCCESS: NLTK resource '{name}' found (using path_suffix: '{path_suffix}').")
        except LookupError:
            messages.append(f"INFO: NLTK resource '{name}' not found using suffix '{path_suffix}'. Attempting to download package ID '{package_id}' to '{app_nltk_data_path}'...")
            try:
                nltk.download(package_id, download_dir=str(app_nltk_data_path), quiet=False)

                # ✅ Patch handling for known 'wordnet' location bug
                if name == "wordnet":
                    wordnet_src = app_nltk_data_path / 'wordnet'
                    wordnet_expected = app_nltk_data_path / 'corpora' / 'wordnet'
                    if wordnet_src.exists():
                        (app_nltk_data_path / 'corpora').mkdir(exist_ok=True)
                        if wordnet_expected.exists():
                            shutil.rmtree(wordnet_expected)
                        shutil.move(str(wordnet_src), str(wordnet_expected))
                        messages.append(f"INFO: Moved 'wordnet' from '{wordnet_src}' to '{wordnet_expected}' to fix NLTK lookup path.")

                nltk.data.find(path_suffix)
                messages.append(f"SUCCESS: NLTK resource '{name}' downloaded and verified in '{app_nltk_data_path}'.")
            except LookupError:
                messages.append(f"ERROR: NLTK resource '{name}' STILL NOT FOUND after forced move and download attempt. NLTK searched in: {nltk.data.path}")
                all_good = False
            except Exception as e_nltk_dl:
                messages.append(f"ERROR: Failed to execute nltk.download for '{name}'. Exception: {e_nltk_dl}")
                all_good = False

    st.session_state.nltk_messages_for_sidebar = messages
    st.session_state.nltk_resources_all_good = all_good

    if all_good:
        st.session_state.nltk_messages_for_sidebar.append("SUCCESS: All required NLTK resources are correctly set up.")
    else:
        st.session_state.nltk_messages_for_sidebar.append("ERROR: One or more NLTK resources could not be set up. Please check the messages above for details.")

    return all_good


NLTK_READY = ensure_nltk_resources()

# --- Import UserTopicModelingSystem ---
nltk_stopwords_global = None 
if NLTK_READY and MODULES_AVAILABLE: # Check MODULES_AVAILABLE here too
    try:
        from user_topic_modeling import UserTopicModelingSystem 
        from nltk.corpus import stopwords 
        nltk_stopwords_global = stopwords 
        USER_TOPIC_MODELING_CLASS_LOADED = True
    except ImportError as e: USER_TOPIC_MODELING_ERROR_MESSAGE = f"Could not import UserTopicModelingSystem class from 'user_topic_modeling.py'. Error: {e}"
    except Exception as e_other: USER_TOPIC_MODELING_ERROR_MESSAGE = f"Unexpected error importing UserTopicModelingSystem class. Error: {e_other}"
else:
    if not NLTK_READY: USER_TOPIC_MODELING_ERROR_MESSAGE = "NLTK resources failed. Topic modeling unavailable."
    elif not MODULES_AVAILABLE: USER_TOPIC_MODELING_ERROR_MESSAGE = "App component files missing. Cannot proceed with topic modeling."


# --- Initialize Session State Variables ---
# (Ensure all relevant session state variables are initialized)
keys_to_init = ['processed_df', 'temp_output_dir_lda', 'viewing_dashboard', 
                'show_detailed_dashboard_button', 'lda_num_topics_run', 
                'error_shown_once_main_import', 'df_original_loaded', 
                'text_col_name', 'user_id_col_name', 'datetime_col_name', 
                'cleaned_text_col_name', 'lda_system_instance', 'sidebar_uploaded_file',
                'nltk_messages_displayed_once'] # Add this for NLTK messages

defaults = {
    'processed_df': None, 'temp_output_dir_lda': None, 'viewing_dashboard': False,
    'show_detailed_dashboard_button': False, 'lda_num_topics_run': 10,
    'error_shown_once_main_import': False, 'df_original_loaded': None,
    'text_col_name': None, 'user_id_col_name': None, 'datetime_col_name': None,
    'cleaned_text_col_name': None, 'lda_system_instance': None, 'sidebar_uploaded_file': None,
    'nltk_messages_displayed_once': False
}
for key, default_val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val


# --- Main App UI ---
st.title("📄 Social Media Text Processor & Topic Analyzer")

if not USER_TOPIC_MODELING_CLASS_LOADED and MODULES_AVAILABLE and not st.session_state.error_shown_once_main_import:
    st.error(USER_TOPIC_MODELING_ERROR_MESSAGE)
    st.session_state.error_shown_once_main_import = True
elif not MODULES_AVAILABLE and not st.session_state.error_shown_once_main_import: # If core modules failed
    # The error from module import try-except is already displayed by Streamlit implicitly or by our st.error
    pass 
st.markdown("---")

# --- Sidebar Rendering ---
# Initialize local variables that will be set/used by sidebar components
df_original_main_scope = None 
user_id_col_main_scope = None
text_col_main_scope = None
datetime_col_main_scope = None
cleaning_opts_main_scope = {}
lda_params_main_scope = {}

if MODULES_AVAILABLE:
    st.sidebar.header("1. Upload Data & Select Columns")
    # File uploader directly in main_app.py
    uploaded_file_main_scope = st.sidebar.file_uploader("Upload your CSV file", type=["csv"], key="main_app_file_uploader")
    st.session_state.sidebar_uploaded_file = uploaded_file_main_scope # Store for potential use by ui_components if needed

    display_nltk_messages_from_session() 

    if uploaded_file_main_scope is not None:
        try:
            st.session_state.df_original_loaded = pd.read_csv(uploaded_file_main_scope)
            st.sidebar.success("File Uploaded Successfully!")
            
            # render_sidebar_setup (from ui_components) uses st.session_state.df_original_loaded
            # and sets user_id_col_name, text_col_name, datetime_col_name in session_state
            render_sidebar_setup(st.session_state.df_original_loaded) 

            # Update local scope variables from session_state
            df_original_main_scope = st.session_state.df_original_loaded
            user_id_col_main_scope = st.session_state.get("user_id_col_name")
            text_col_main_scope = st.session_state.get("text_col_name")
            datetime_col_main_scope = st.session_state.get("datetime_col_name")

            if text_col_main_scope:
                if df_original_main_scope is not None:
                    # This preview is shown when a file is loaded and text column selected
                    # It will be overwritten if "Process Data" is clicked or dashboard is viewed
                    if not st.session_state.get('viewing_dashboard') and st.session_state.get('processed_df') is None:
                        st.header("Original Data Preview (First 5 rows)") 
                        st.dataframe(df_original_main_scope.head())
            
        except Exception as e:
            st.sidebar.error(f"Error reading or processing file: {e}")
            st.session_state.df_original_loaded = None 
            df_original_main_scope = None 

    # Cleaning Options
    if st.session_state.get("df_original_loaded") is not None and st.session_state.get("text_col_name"):
        st.sidebar.markdown("---")
        st.sidebar.header("2. Cleaning Options")
        cleaning_opts_main_scope = render_cleaning_options_sidebar() 
        
        if st.sidebar.button("✨ Process Data", key="process_data_main_app_button_final"):
            if st.session_state.text_col_name in st.session_state.df_original_loaded.columns:
                with st.spinner("Processing text data..."):
                    st.session_state.processed_df = perform_text_cleaning(
                        st.session_state.df_original_loaded.copy(), 
                        st.session_state.text_col_name, 
                        cleaning_opts_main_scope,
                        nltk_stopwords_module_passed=nltk_stopwords_global 
                    )
                    # cleaned_text_col_name is set in session_state by perform_text_cleaning
                st.success("Text processing complete!")
                # The preview of processed_df will be handled in the main content area logic
            else:
                st.sidebar.error(f"Selected text column '{st.session_state.text_col_name}' not found.")
    
    # LDA Parameters and Run Button
    if USER_TOPIC_MODELING_CLASS_LOADED:
        st.sidebar.markdown("---")
        st.sidebar.header("3. LDA Topic Modeling")
        lda_params_main_scope = render_lda_parameters_sidebar()
    
        if st.session_state.get('processed_df') is not None:
            if st.sidebar.button("🚀 Run LDA Topic Modeling", key="run_lda_main_app_button_final"):
                if not all([st.session_state.get("user_id_col_name"), 
                            st.session_state.get("text_col_name"), 
                            st.session_state.get("datetime_col_name")]):
                    st.error("User ID, Original Text, and Datetime columns must be selected for LDA.")
                elif st.session_state.get("cleaned_text_col_name") not in st.session_state.processed_df.columns:
                     st.error("Cleaned text column not found. Please process data first.")
                else:
                    overall_success_lda, lda_results_summary_main = run_lda_pipeline_with_checklist(
                        processed_df=st.session_state.processed_df,
                        user_id_col_name=st.session_state.user_id_col_name,
                        text_col_name_original=st.session_state.text_col_name, 
                        cleaned_text_col_name=st.session_state.cleaned_text_col_name,
                        datetime_col_name=st.session_state.datetime_col_name,
                        lda_params=lda_params_main_scope # Pass the dict of params
                    )
                    if overall_success_lda:
                        st.session_state.show_detailed_dashboard_button = True
                        st.session_state.lda_num_topics_run = lda_params_main_scope['num_topics']
                        # Summary from orchestrator is displayed by the checklist itself now.
                    else:
                        st.session_state.show_detailed_dashboard_button = False
        elif st.session_state.get("df_original_loaded") is not None and st.session_state.get("text_col_name"):
            st.sidebar.info("Click 'Process Data' before running LDA Topic Modeling.")
    elif MODULES_AVAILABLE: 
        st.sidebar.warning("LDA Topic Modeling system could not be loaded. Check errors at the top of the page.")


# --- Main Content Area: Dashboard or Initial/Processed Data View ---
if st.session_state.get('viewing_dashboard'):
    if st.session_state.get('temp_output_dir_lda') and MODULES_AVAILABLE and USER_TOPIC_MODELING_CLASS_LOADED:
        render_full_dashboard(st.session_state.temp_output_dir_lda)
    else:
        if not MODULES_AVAILABLE: st.error("Dashboard components could not be loaded.")
        elif not USER_TOPIC_MODELING_CLASS_LOADED: st.error("Topic modeling system not loaded. Dashboard unavailable.")
        else: st.error("LDA results directory not found. Please run the LDA pipeline first.")
        
        if st.button("Go to Setup Page", key="dashboard_back_setup_btn_app_main_final"):
            st.session_state.viewing_dashboard = False
            st.session_state.show_detailed_dashboard_button = False
            st.rerun()
else: 
    # Display initial messages OR processed data preview if LDA not run yet OR dashboard button
    if st.session_state.get('processed_df') is not None:
        # This preview is shown after "Process Data" is clicked, if not viewing dashboard
        # and if LDA hasn't been run yet to show its checklist
        if not st.session_state.get('show_detailed_dashboard_button'): # Only show if LDA summary/checklist isn't active
            st.subheader("Processed Data Preview")
            processed_df_display = st.session_state.processed_df
            cols_to_show_main_area = [
                col for col in [
                    st.session_state.get("user_id_col_name"), 
                    st.session_state.get("datetime_col_name"), 
                    st.session_state.get("text_col_name"), 
                    st.session_state.get("cleaned_text_col_name"), 
                    'extracted_hashtags', 
                    'extracted_mentions'
                ] if col and col in processed_df_display.columns
            ]
            if cols_to_show_main_area: st.dataframe(processed_df_display[cols_to_show_main_area].head())
            else: st.warning("No relevant columns to show in processed data preview (main area).")
            
            @st.cache_data
            def convert_df_to_csv_main_processed_area_final(df_to_convert): return df_to_convert.to_csv(index=False).encode('utf-8')
            csv_output_main_proc_area = convert_df_to_csv_main_processed_area_final(processed_df_display)
            st.download_button(label="📥 Download Processed CSV", data=csv_output_main_proc_area, file_name="cleaned_data.csv", mime="text/csv", key="dl_cleaned_main_area_btn_final")

    elif st.session_state.get("df_original_loaded") is None and MODULES_AVAILABLE: 
        # Show only if no critical import error message is already displayed by USER_TOPIC_MODELING_CLASS_LOADED check
        if USER_TOPIC_MODELING_CLASS_LOADED or not st.session_state.error_shown_once_main_import:
             st.info("☝️ Please upload a CSV file using the sidebar to begin.")
    elif st.session_state.get("df_original_loaded") is not None and not st.session_state.get("text_col_name") and MODULES_AVAILABLE:
         st.warning("👈 Please select the 'Text/Content Column' in the sidebar to proceed.")
    
    # Button to view dashboard (appears after successful LDA run)
    if st.session_state.get('show_detailed_dashboard_button') and MODULES_AVAILABLE and USER_TOPIC_MODELING_CLASS_LOADED:
        st.markdown("---") 
        if st.button("📊 View Detailed Analysis Dashboard", key="view_dashboard_btn_page_app_main_final"):
            st.session_state.viewing_dashboard = True
            st.session_state.show_detailed_dashboard_button = False 
            st.rerun()

# --- Footer and Manual Cleanup Button (always in sidebar) ---
if MODULES_AVAILABLE:
    if 'temp_output_dir_lda' in st.session_state and st.session_state.temp_output_dir_lda:
        st.sidebar.markdown("---")
        if st.sidebar.button("Manually Cleanup LDA Temp Directory", key="cleanup_btn_sidebar_app_main_final"):
            temp_dir_to_clean = st.session_state.temp_output_dir_lda
            if temp_dir_to_clean and os.path.exists(temp_dir_to_clean):
                try:
                    shutil.rmtree(temp_dir_to_clean)
                    st.sidebar.success(f"Successfully removed: {temp_dir_to_clean}")
                    del st.session_state.temp_output_dir_lda
                    st.session_state.viewing_dashboard = False 
                    st.session_state.show_detailed_dashboard_button = False
                    if 'lda_system_instance' in st.session_state: del st.session_state.lda_system_instance
                    st.rerun()
                except Exception as e: st.sidebar.error(f"Error removing {temp_dir_to_clean}: {e}")
            else: st.sidebar.info("No active LDA temporary directory or path does not exist.")

st.markdown("---")
st.markdown("CIVICTECH DC - CIB Mango Tree")
