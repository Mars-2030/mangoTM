# lda_pipeline_orchestrator.py
import streamlit as st
import pandas as pd
import numpy as np
import os
from pathlib import Path
import tempfile
import shutil # For cleaning up temp dir if absolutely necessary here (usually managed by main_app)
import traceback
import io
import sys
import json # For saving coherence score
import pickle
# This module will need UserTopicModelingSystem
# Ensure user_topic_modeling.py is in the Python path
try:
    from user_topic_modeling import UserTopicModelingSystem
    # Also need CoherenceModel if calculating it here
    from gensim.models.coherencemodel import CoherenceModel
except ImportError as e:
    # This error should ideally be caught and displayed by main_app.py
    # If this module is imported, it's assumed UserTopicModelingSystem is available.
    print(f"CRITICAL ERROR in lda_pipeline_orchestrator: Could not import UserTopicModelingSystem or CoherenceModel: {e}")
    # A more robust way would be to have main_app.py pass the class or instance if needed.
    # For now, direct import is assumed to work based on main_app's successful import.
    UserTopicModelingSystem = None # Define it as None to prevent further errors if import failed
    CoherenceModel = None


# --- Helper Class for Streamlit Output ---
class StreamlitLogHandler(io.StringIO):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self._buffer = [] 

    def write(self, s):
        if s.strip(): 
            self._buffer.append(s) 
            log_content = "".join(self._buffer)
            self.placeholder.code(log_content.strip(), language='text')

    def flush(self):
        pass


def run_lda_pipeline_with_checklist(
    processed_df: pd.DataFrame,
    user_id_col_name: str,
    text_col_name_original: str, # Original text column name before cleaning
    cleaned_text_col_name: str,  # Name of the column with cleaned text
    datetime_col_name: str,
    lda_params: dict # Dictionary from render_lda_parameters_sidebar
):
    """
    Orchestrates the LDA topic modeling pipeline with detailed Streamlit checklist logging.
    """
    st.markdown("---")
    st.subheader("LDA Pipeline Progress:")
    pipeline_steps_config = {
        "init_system": {"title": "Initializing LDA System", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "load_data": {"title": "Loading data into system", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "preprocess_data": {"title": "Preprocessing data for LDA", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "create_lda_model": {"title": "Creating LDA model", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "get_doc_topics": {"title": "Calculating document topics", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "user_topic_dist": {"title": "Calculating user topic distributions", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "user_narrowness": {"title": "Calculating user narrowness scores", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "temporal_data": {"title": "Calculating temporal topic data", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "coherence": {"title": "Calculating topic coherence", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "suspicious_users": {"title": "Detecting suspicious users", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "export_data": {"title": "Exporting results", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
        "visualizations": {"title": "Generating & Saving Visualizations", "log_placeholder": st.empty(), "status_placeholder": st.empty()},
    }
    for key, config in pipeline_steps_config.items():
        config["status_placeholder"].markdown(f"⚪ Pending: {config['title']}")
    
    overall_success = True
    lda_results_summary = {}
    lda_system_instance = None
    original_stdout = sys.stdout

    # Check if UserTopicModelingSystem was imported successfully
    if UserTopicModelingSystem is None:
        st.error("UserTopicModelingSystem class is not available. Cannot run LDA pipeline.")
        return False, {"error": "UserTopicModelingSystem import failed"}

    try:
        # --- 0. Prepare data for UserTopicModelingSystem ---
        df_for_lda_pipeline = processed_df[[
            user_id_col_name, datetime_col_name, cleaned_text_col_name
        ]].copy()
        df_for_lda_pipeline.rename(columns={
            user_id_col_name: 'user_id',
            datetime_col_name: 'timestamp',
            cleaned_text_col_name: 'post_content'
        }, inplace=True)
        df_for_lda_pipeline['timestamp'] = pd.to_datetime(df_for_lda_pipeline['timestamp'], errors='coerce')
        df_for_lda_pipeline = df_for_lda_pipeline.dropna(subset=['user_id', 'timestamp', 'post_content'])
        df_for_lda_pipeline = df_for_lda_pipeline[df_for_lda_pipeline['post_content'].astype(str).str.strip().str.len() > 5]

        if df_for_lda_pipeline.empty:
            st.error("No valid data remaining after preparing for LDA. Check input columns, cleaning, and ensure posts are not too short.")
            raise ValueError("No data for LDA pipeline")

        # Manage temporary directory (main_app.py stores this in session_state)
        if 'temp_output_dir_lda' in st.session_state and st.session_state.temp_output_dir_lda and os.path.exists(st.session_state.temp_output_dir_lda):
            shutil.rmtree(st.session_state.temp_output_dir_lda, ignore_errors=True)
        temp_output_dir_lda = tempfile.mkdtemp(prefix="lda_streamlit_")
        st.session_state.temp_output_dir_lda = temp_output_dir_lda # Crucial for main_app to find results
        
        temp_input_csv_lda = os.path.join(temp_output_dir_lda, "temp_lda_input.csv")
        df_for_lda_pipeline.to_csv(temp_input_csv_lda, index=False)
        
        def execute_step_local(step_key_exec, function_to_call, *args, **kwargs):
            nonlocal overall_success # To modify the flag in the outer function's scope
            if not overall_success: 
                pipeline_steps_config[step_key_exec]["status_placeholder"].markdown(f"⚪ Skipped (due to previous error): {pipeline_steps_config[step_key_exec]['title']}")
                return None
            config = pipeline_steps_config[step_key_exec]
            config["status_placeholder"].markdown(f"⏳ In Progress: {config['title']}...")
            log_handler = StreamlitLogHandler(config["log_placeholder"])
            sys.stdout = log_handler
            try:
                result = function_to_call(*args, **kwargs)
                sys.stdout = original_stdout
                config["status_placeholder"].markdown(f"✅ Done: {config['title']}")
                return result
            except Exception as e_step:
                sys.stdout = original_stdout
                config["status_placeholder"].markdown(f"❌ Failed: {config['title']}")
                config["log_placeholder"].error(f"Error in step '{config['title']}': {e_step}\n{traceback.format_exc()}")
                overall_success = False
                raise 
        
        # Pipeline steps
        extra_stops_list = [s.strip().lower() for s in lda_params.get('extra_stopwords_lda', '').split(',') if s.strip()]
        lda_system_instance = execute_step_local(
            "init_system", UserTopicModelingSystem,
            num_topics=lda_params['num_topics'], time_bin=lda_params['time_bin'], lemmatize=True,
            extra_stopwords=extra_stops_list, min_post_length=lda_params['min_post_length']
        )
        if not lda_system_instance: raise ValueError("System Initialization Failed")
        st.session_state.lda_system_instance = lda_system_instance # Store for potential dashboard use
        
        execute_step_local("load_data", lda_system_instance.load_data, temp_input_csv_lda)
        if lda_system_instance.data is None or lda_system_instance.data.empty: raise ValueError("Data loading in system failed.")
        lda_results_summary['initial_records_in_system'] = len(lda_system_instance.data)

        execute_step_local("preprocess_data", lda_system_instance.preprocess_data, combine_by_window=lda_params['combine_posts'])
        if lda_system_instance.documents is None or not lda_system_instance.documents: raise ValueError("Preprocessing failed.")
        lda_results_summary['records_after_preprocessing_in_system'] = len(lda_system_instance.data)
        lda_results_summary['documents_for_lda'] = len(lda_system_instance.documents)

        execute_step_local("create_lda_model", lda_system_instance.create_lda_model, lda_system_instance.documents)
        if lda_system_instance.lda_model is None: raise ValueError("LDA model creation failed.")
        
        if execute_step_local("get_doc_topics", lda_system_instance.get_document_topics) is None: raise ValueError("Doc topic calculation failed.")
        
        if execute_step_local("user_topic_dist", lda_system_instance.calculate_user_topic_distributions) is None: st.warning("User distributions calculation might have issues.")
        lda_results_summary['num_users_processed'] = len(lda_system_instance.user_topic_distributions) if lda_system_instance.user_topic_distributions else 0
        
        if execute_step_local("user_narrowness", lda_system_instance.calculate_user_narrowness_scores) is None: st.warning("Narrowness scores calculation might have issues.")
        
        # Temporal data calculation AND saving (as UserTopicModelingSystem was modified)
        if execute_step_local("temporal_data", lda_system_instance.calculate_temporal_topic_data) is None: 
            st.warning("Temporal data calculation might have issues.")
        # The saving of temporal_topic_data.pkl is now part of UserTopicModelingSystem's run_full_pipeline
        # If we are calling methods individually, we need to replicate that saving here,
        # OR modify UserTopicModelingSystem to have a separate save_temporal_data method.
        # For now, assume calculate_temporal_topic_data populates self.temporal_topic_data
        # and we save it here.
        if overall_success and lda_system_instance.temporal_topic_data:
            pipeline_steps_config["temporal_data"]["log_placeholder"].code("   Attempting to save temporal data...")
            temporal_pickle_path = Path(temp_output_dir_lda) / "temporal_topic_data.pkl"
            try:
                with open(temporal_pickle_path, 'wb') as f_pkl:
                    pickle.dump(lda_system_instance.temporal_topic_data, f_pkl)
                pipeline_steps_config["temporal_data"]["log_placeholder"].code(f"   Temporal topic data saved to {temporal_pickle_path}")
                lda_results_summary['temporal_data_saved_path'] = str(temporal_pickle_path)
            except Exception as e_pickle_orch:
                pipeline_steps_config["temporal_data"]["log_placeholder"].error(f"   Error saving temporal data: {e_pickle_orch}")


        # Coherence Score
        if overall_success and CoherenceModel is not None:
            pipeline_steps_config["coherence"]["status_placeholder"].markdown(f"⏳ In Progress: {pipeline_steps_config['coherence']['title']}...")
            coherence_score_cv = float('nan')
            log_handler_coherence = StreamlitLogHandler(pipeline_steps_config["coherence"]["log_placeholder"])
            sys.stdout = log_handler_coherence
            try:
                if lda_system_instance.lda_model and lda_system_instance.dictionary and lda_system_instance.documents:
                    valid_docs_for_coherence = [doc for doc in lda_system_instance.documents if doc]
                    if valid_docs_for_coherence and lda_system_instance.corpus:
                        print("   Calculating c_v coherence...")
                        coherence_model_lda = CoherenceModel(model=lda_system_instance.lda_model, texts=valid_docs_for_coherence, dictionary=lda_system_instance.dictionary, coherence='c_v')
                        coherence_score_cv = coherence_model_lda.get_coherence()
                        lda_results_summary['coherence_score_c_v'] = coherence_score_cv
                        print(f"   LDA Coherence Score (c_v): {coherence_score_cv:.4f}")
                    else: print("   Warning: No valid documents/corpus for coherence.")
                else: print("   Warning: LDA model, dictionary, or documents not available for coherence.")
                sys.stdout = original_stdout
                pipeline_steps_config["coherence"]["status_placeholder"].markdown(f"✅ Done: {pipeline_steps_config['coherence']['title']} (c_v: {coherence_score_cv:.4f})")
            except Exception as e_coh:
                sys.stdout = original_stdout; overall_success = False
                pipeline_steps_config["coherence"]["status_placeholder"].markdown(f"❌ Failed: {pipeline_steps_config['coherence']['title']}")
                pipeline_steps_config["coherence"]["log_placeholder"].error(f"Error in coherence: {e_coh}\n{traceback.format_exc()}")
            coherence_path = Path(temp_output_dir_lda) / "topic_coherence.json"
            with open(coherence_path, 'w') as f: json.dump({'coherence_mean': coherence_score_cv if pd.notna(coherence_score_cv) else None}, f, indent=2)
        elif not CoherenceModel:
             pipeline_steps_config["coherence"]["status_placeholder"].markdown(f"⚪ Skipped: CoherenceModel not available.")
        else: # Skipped due to previous error
             pipeline_steps_config["coherence"]["status_placeholder"].markdown(f"⚪ Skipped (due to previous error): {pipeline_steps_config['coherence']['title']}")

        if lda_params.get('run_suspicious_detection', True): # Check against lda_params
            suspicious_df = execute_step_local("suspicious_users", lda_system_instance.detect_suspicious_users)
            if suspicious_df is not None and overall_success: # Check overall_success again
                suspicious_df.to_csv(Path(temp_output_dir_lda) / "suspicious_users.csv", index=False)
                lda_results_summary['suspicious_users_detected'] = suspicious_df['suspicious'].sum()
                # Status already updated by execute_step_local, but we can add count
                pipeline_steps_config["suspicious_users"]["status_placeholder"].markdown(f"✅ Done: {pipeline_steps_config['suspicious_users']['title']} ({lda_results_summary.get('suspicious_users_detected', 0)} found)")
        else:
            pipeline_steps_config["suspicious_users"]["status_placeholder"].markdown(f"⚪ Skipped: Detecting suspicious users (user disabled)")

        if overall_success: 
            execute_step_local("export_data", lambda: (
                lda_system_instance.export_user_topic_data(Path(temp_output_dir_lda) / "user_topic_data.csv"),
                lda_system_instance.export_topic_words(Path(temp_output_dir_lda) / "topic_words.csv")
            ))
        else:
            pipeline_steps_config["export_data"]["status_placeholder"].markdown(f"⚪ Skipped (due to previous error): {pipeline_steps_config['export_data']['title']}")

        if lda_params.get('run_visualizations', True) and overall_success:
            pipeline_steps_config["visualizations"]["status_placeholder"].markdown(f"⏳ In Progress: {pipeline_steps_config['visualizations']['title']}...")
            log_handler_viz = StreamlitLogHandler(pipeline_steps_config["visualizations"]["log_placeholder"])
            sys.stdout = log_handler_viz
            try:
                lda_system_instance.visualize_topic_words(save_path=Path(temp_output_dir_lda) / "topic_words.png")
                lda_system_instance.visualize_narrowness_vs_frequency(save_path=Path(temp_output_dir_lda) / "narrowness_vs_frequency.png")
                lda_system_instance.visualize_topic_embedding(save_path=Path(temp_output_dir_lda) / "topic_embedding.png")
                if lda_system_instance.user_narrowness_scores is not None and not lda_system_instance.user_narrowness_scores.empty:
                    top_users_for_viz = lda_system_instance.user_narrowness_scores.sort_values('gini_coefficient', ascending=False).head(3)['user_id'].astype(str).tolist()
                    for user_id_v in top_users_for_viz:
                        lda_system_instance.visualize_user_topic_pie(user_id_v, save_path=Path(temp_output_dir_lda) / f"user_{user_id_v}_topics.png")
                        lda_system_instance.visualize_user_temporal_topics(user_id_v, save_path=Path(temp_output_dir_lda) / f"user_{user_id_v}_temporal.png")
                sys.stdout = original_stdout
                pipeline_steps_config["visualizations"]["status_placeholder"].markdown(f"✅ Done: {pipeline_steps_config['visualizations']['title']}")
            except Exception as e_viz:
                sys.stdout = original_stdout; overall_success = False # Mark as failure if viz fails
                pipeline_steps_config["visualizations"]["status_placeholder"].markdown(f"❌ Failed: {pipeline_steps_config['visualizations']['title']}")
                pipeline_steps_config["visualizations"]["log_placeholder"].error(f"Error in visualizations: {e_viz}\n{traceback.format_exc()}")
        elif not overall_success:
             pipeline_steps_config["visualizations"]["status_placeholder"].markdown(f"⚪ Skipped (due to previous error): {pipeline_steps_config['visualizations']['title']}")
        else: 
            pipeline_steps_config["visualizations"]["status_placeholder"].markdown(f"⚪ Skipped: Generating visualizations (user disabled)")
        
        lda_results_summary['num_topics_configured'] = lda_params['num_topics']
        lda_results_summary['output_directory_temp'] = str(temp_output_dir_lda)
    
    except Exception as e:
        overall_success = False 
        # This will catch errors from execute_step_local if they are re-raised,
        # or from the initial data prep before execute_step_local calls.
        st.error(f"A critical error occurred in the LDA pipeline: {e}")
        st.text(traceback.format_exc())
    finally:
        sys.stdout = original_stdout # Ensure stdout is always restored

    return overall_success, lda_results_summary