# comparison_script.py

import pandas as pd
import numpy as np  # <-- Import NumPy
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import os
import pickle
from datetime import datetime # Added for report generation

# Import matplotlib_venn if available, otherwise skip venn diagram
try:
    from matplotlib_venn import venn2
    has_venn = True
except ImportError:
    has_venn = False

# Import scipy.stats if available for regression line
try:
    from scipy import stats
    has_scipy = True
except ImportError:
    has_scipy = False


from user_topic_modeling import UserTopicModelingSystem
from bertopic_extension import BERTopicModelingSystem


# --- Helper function for JSON serialization ---
def handle_numpy_types(obj):
    """
    Convert numpy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # Handle numpy arrays if they accidentally get in structures
        return obj.tolist()
    # Let the default encoder handle other types or raise errors
    # Raising TypeError here ensures non-serializable types are still caught
    # Use default=str as a fallback ONLY if absolutely needed, but raising TypeError is safer
    # return str(obj) # <-- Fallback if you MUST serialize unknown types, but likely hides issues
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable by default handler")
# --- End Helper function ---


def run_comparison(input_file: str, output_dir: str,
                  lda_topics: int = 20, bert_topics: int = None, # Changed default bert_topics to None
                  combine_by_window: bool = True):
    """
    Run both LDA and BERTopic models on the same dataset and compare results.

    Args:
        input_file: Path to input data file
        output_dir: Directory to save comparison outputs
        lda_topics: Number of topics for LDA model
        bert_topics: Target number of topics for BERTopic model (or None to auto-detect)
        combine_by_window: Whether to combine posts into time windows
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create subdirectories for model outputs
    lda_dir = os.path.join(output_dir, "lda_results")
    bert_dir = os.path.join(output_dir, "bertopic_results")
    comparison_out_dir = os.path.join(output_dir, "model_comparison") # Specific dir for comparison outputs
    os.makedirs(lda_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    os.makedirs(comparison_out_dir, exist_ok=True)

    # Step 1: Run LDA model
    print("\nRunning LDA model...")
    # Check if results already exist
    lda_user_data_path = os.path.join(lda_dir, "user_topic_data.csv")
    if os.path.exists(lda_user_data_path):
         print("LDA results found, skipping LDA run.")
         # Dummy results structure if needed later
         lda_results = {'output_dir': lda_dir}
    else:
        lda_system = UserTopicModelingSystem(
            num_topics=lda_topics,
            time_bin='week', # Assuming week, adjust if needed or make parameter
            lemmatize=True,
            extra_stopwords=['rt', 'http', 'https', 'amp'] # Example stopwords
        )

        lda_results = lda_system.run_full_pipeline(
            input_file=input_file,
            output_dir=lda_dir,
            combine_by_window=combine_by_window,
            visualize_topics=True,
            detect_suspicious=True
        )
        # Save LDA config
        lda_config = {'num_topics': lda_topics, 'combine_by_window': combine_by_window, 'lemmatize': True}
        try:
            with open(os.path.join(lda_dir, "config.json"), 'w') as f:
                json.dump(lda_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save LDA config: {e}")


    # Step 2: Run BERTopic model
    print("\nRunning BERTopic model...")
    # Check if results already exist
    bert_user_data_path = os.path.join(bert_dir, "user_topic_data.csv")
    if os.path.exists(bert_user_data_path):
         print("BERTopic results found, skipping BERTopic run.")
         # Dummy results structure if needed later
         bert_results = {'output_dir': bert_dir}
    else:
        bert_system = BERTopicModelingSystem(
            language_model='all-MiniLM-L6-v2',
            time_bin='week', # Assuming week, adjust if needed or make parameter
            min_topic_size=10, # Example min_topic_size
            extra_stopwords=['rt', 'http', 'https', 'amp'], # Example stopwords
            n_neighbors=15,
            n_components=5,
            min_cluster_size=10, # Usually tied to min_topic_size
            num_topics=bert_topics # Pass the target number of topics
        )

        bert_results = bert_system.run_full_pipeline(
            input_file=input_file,
            output_dir=bert_dir,
            combine_by_window=combine_by_window,
            visualize_topics=True,
            detect_suspicious=True
        )
        # Save BERTopic config
        bert_config = {'num_topics': bert_topics, 'min_topic_size': 10, 'combine_by_window': combine_by_window}
        try:
            with open(os.path.join(bert_dir, "config.json"), 'w') as f:
                 json.dump(bert_config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save BERTopic config: {e}")


    # Step 3: Compare results
    print("\nComparing results...")
    compare_model_results(lda_dir, bert_dir, comparison_out_dir) # Save comparison to its own folder

    return {
        "lda_results": lda_results,
        "bert_results": bert_results
    }


def compare_model_results(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Compare the results of LDA and BERTopic models.

    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs
    """
    print(f"Comparison outputs will be saved to: {output_dir}")
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Paths to data files
    lda_user_data_path = os.path.join(lda_dir, "user_topic_data.csv")
    bert_user_data_path = os.path.join(bert_dir, "user_topic_data.csv")
    lda_suspicious_path = os.path.join(lda_dir, "suspicious_users.csv")
    bert_suspicious_path = os.path.join(bert_dir, "suspicious_users.csv")

    # Load user topic data - check if files exist
    if not os.path.exists(lda_user_data_path):
        print(f"Warning: LDA user data not found at {lda_user_data_path}. Skipping narrowness comparison.")
        lda_user_data = None
    else:
        lda_user_data = pd.read_csv(lda_user_data_path)

    if not os.path.exists(bert_user_data_path):
        print(f"Warning: BERTopic user data not found at {bert_user_data_path}. Skipping narrowness comparison.")
        bert_user_data = None
    else:
        bert_user_data = pd.read_csv(bert_user_data_path)

    # 1. Compare user narrowness metrics (only if both loaded)
    if lda_user_data is not None and bert_user_data is not None:
        print("Comparing narrowness metrics...")
        compare_narrowness_metrics(lda_user_data, bert_user_data, output_dir)
    else:
         print("Skipping narrowness comparison due to missing data.")

    # Load suspicious users data - check if files exist
    if not os.path.exists(lda_suspicious_path):
         print(f"Warning: LDA suspicious users data not found at {lda_suspicious_path}. Skipping suspicious comparison.")
         lda_suspicious = None
    else:
         lda_suspicious = pd.read_csv(lda_suspicious_path)

    if not os.path.exists(bert_suspicious_path):
         print(f"Warning: BERTopic suspicious users data not found at {bert_suspicious_path}. Skipping suspicious comparison.")
         bert_suspicious = None
    else:
         bert_suspicious = pd.read_csv(bert_suspicious_path)

    # 2. Compare suspicious user detection (only if both loaded)
    if lda_suspicious is not None and bert_suspicious is not None:
        print("Comparing suspicious user detection...")
        compare_suspicious_detection(lda_suspicious, bert_suspicious, output_dir)
    else:
         print("Skipping suspicious user comparison due to missing data.")

    # 3. Compare topic coherence (if available)
    print("Comparing topic coherence (if available)...")
    topic_coherence_comparison(lda_dir, bert_dir, output_dir)

    # 4. Compare topic words (qualitative analysis)
    print("Comparing topic words...")
    topic_words_comparison(lda_dir, bert_dir, output_dir)

    # 5. Generate comparison report
    print("Generating comparison report...")
    generate_comparison_report(lda_dir, bert_dir, output_dir)


def compare_narrowness_metrics(lda_data: pd.DataFrame, bert_data: pd.DataFrame, output_dir: str):
    """
    Compare user narrowness metrics between LDA and BERTopic.

    Args:
        lda_data: DataFrame with LDA user topic data
        bert_data: DataFrame with BERTopic user topic data
        output_dir: Directory to save comparison outputs
    """
    # Merge data on user_id
    merged_data = lda_data[['user_id', 'post_count', 'gini_coefficient', 'shannon_entropy', 'top1_ratio']].merge(
        bert_data[['user_id', 'post_count', 'gini_coefficient', 'shannon_entropy', 'top1_ratio']],
        on='user_id',
        suffixes=('_lda', '_bert'),
        how='inner' # Use inner join to compare only users present in both results
    )

    if merged_data.empty:
        print("Warning: No common users found between LDA and BERTopic results for narrowness comparison.")
        return None

    # Create correlation plots
    metrics = ['gini_coefficient', 'shannon_entropy', 'top1_ratio']

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('User Narrowness Metrics Comparison: LDA vs BERTopic', fontsize=16)

    for i, metric in enumerate(metrics):
        lda_metric = f"{metric}_lda"
        bert_metric = f"{metric}_bert"

        # Check if metric columns exist after merge
        if lda_metric not in merged_data.columns or bert_metric not in merged_data.columns:
             print(f"Warning: Metric '{metric}' not found in merged data. Skipping plot.")
             continue

        # Calculate correlation
        # Handle potential NaN values before calculating correlation
        valid_data = merged_data[[lda_metric, bert_metric]].dropna()
        if valid_data.empty or len(valid_data) < 2:
             print(f"Warning: Not enough valid data points for metric '{metric}' correlation.")
             corr = np.nan
             r_value = np.nan
        else:
             corr = valid_data[lda_metric].corr(valid_data[bert_metric])
             # Calculate regression line if scipy is available
             if has_scipy:
                 try:
                     slope, intercept, r_value, p_value, std_err = stats.linregress(
                         valid_data[lda_metric], valid_data[bert_metric]
                     )
                 except ValueError: # Handle cases with insufficient points for linregress
                     slope, intercept, r_value, p_value, std_err = np.nan, np.nan, np.nan, np.nan, np.nan
             else:
                 r_value = np.nan # Assign NaN if scipy is not available


        # Create scatter plot
        axes[i].scatter(merged_data[lda_metric], merged_data[bert_metric], alpha=0.5, label=f'{metric} points')
        axes[i].set_title(f"{metric.replace('_', ' ').title()}\nCorrelation: {corr:.3f}", fontsize=12)
        axes[i].set_xlabel(f"LDA {metric.replace('_', ' ').title()}")
        axes[i].set_ylabel(f"BERTopic {metric.replace('_', ' ').title()}")
        axes[i].grid(True, alpha=0.3)

        # Add diagonal line (line of perfect agreement)
        min_val = min(merged_data[lda_metric].min(), merged_data[bert_metric].min())
        max_val = max(merged_data[lda_metric].max(), merged_data[bert_metric].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')

        # Add regression line if calculated
        if has_scipy and not np.isnan(r_value):
            axes[i].plot(
                [min_val, max_val],
                [intercept + slope * min_val, intercept + slope * max_val],
                'g-',
                label=f'Regression (r²={r_value**2:.3f})'
            )

        axes[i].legend(fontsize=9)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plot_path = os.path.join(output_dir, "narrowness_metrics_comparison.png")
    plt.savefig(plot_path)
    print(f"Narrowness comparison plot saved to {plot_path}")
    plt.close()

    # Calculate summary statistics
    summary = {}
    for metric in metrics:
        lda_metric = f"{metric}_lda"
        bert_metric = f"{metric}_bert"

        if lda_metric not in merged_data.columns or bert_metric not in merged_data.columns:
            continue

        valid_data = merged_data[[lda_metric, bert_metric]].dropna()
        if valid_data.empty or len(valid_data) < 2:
             corr_val = np.nan
        else:
             corr_val = valid_data[lda_metric].corr(valid_data[bert_metric])


        summary[metric] = {
            'correlation': corr_val,
            'lda_mean': merged_data[lda_metric].mean(),
            'bert_mean': merged_data[bert_metric].mean(),
            'lda_std': merged_data[lda_metric].std(),
            'bert_std': merged_data[bert_metric].std(),
            'mean_difference': merged_data[bert_metric].mean() - merged_data[lda_metric].mean()
        }

    # Save summary
    summary_path = os.path.join(output_dir, "narrowness_metrics_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=handle_numpy_types) # Use handler
        print(f"Narrowness summary saved to {summary_path}")
    except Exception as e:
        print(f"Error saving narrowness summary JSON: {e}")

    return merged_data


def compare_suspicious_detection(lda_suspicious: pd.DataFrame, bert_suspicious: pd.DataFrame, output_dir: str):
    """
    Compare suspicious user detection between LDA and BERTopic.

    Args:
        lda_suspicious: DataFrame with LDA suspicious users
        bert_suspicious: DataFrame with BERTopic suspicious users
        output_dir: Directory to save comparison outputs
    """
    # Ensure 'suspicious' column exists
    if 'suspicious' not in lda_suspicious.columns or 'user_id' not in lda_suspicious.columns:
         print("Warning: 'suspicious' or 'user_id' column missing in LDA data. Skipping suspicious comparison.")
         return None
    if 'suspicious' not in bert_suspicious.columns or 'user_id' not in bert_suspicious.columns:
         print("Warning: 'suspicious' or 'user_id' column missing in BERTopic data. Skipping suspicious comparison.")
         return None


    # Get lists of suspicious users
    lda_suspicious_users = set(lda_suspicious[lda_suspicious['suspicious']]['user_id'].astype(str))
    bert_suspicious_users = set(bert_suspicious[bert_suspicious['suspicious']]['user_id'].astype(str))

    # Calculate overlap
    overlap = lda_suspicious_users.intersection(bert_suspicious_users)
    lda_only = lda_suspicious_users - bert_suspicious_users
    bert_only = bert_suspicious_users - lda_suspicious_users

    # Create summary
    num_lda_suspicious = len(lda_suspicious_users)
    num_bert_suspicious = len(bert_suspicious_users)

    summary = {
        'total_suspicious_lda': num_lda_suspicious,
        'total_suspicious_bert': num_bert_suspicious,
        'overlap_count': len(overlap),
        'overlap_percent_of_lda': (len(overlap) / max(1, num_lda_suspicious)) * 100,
        'overlap_percent_of_bert': (len(overlap) / max(1, num_bert_suspicious)) * 100,
        'lda_only_count': len(lda_only),
        'bert_only_count': len(bert_only)
    }

    # Save summary
    summary_path = os.path.join(output_dir, "suspicious_detection_summary.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=handle_numpy_types) # Use handler
        print(f"Suspicious detection summary saved to {summary_path}")
    except Exception as e:
        print(f"Error saving suspicious detection summary JSON: {e}")


    # Create Venn diagram if library is available
    if has_venn:
        plt.figure(figsize=(8, 6))
        v = venn2(
                subsets=(len(lda_only), len(bert_only), len(overlap)),
                set_labels=('LDA Suspicious', 'BERTopic Suspicious')
            )

        # Customize colors and labels if needed
        if v is not None: # Check if venn2 returned a diagram object
            try:
                if v.get_patch_by_id('10'): v.get_patch_by_id('10').set_color('skyblue')
                if v.get_patch_by_id('01'): v.get_patch_by_id('01').set_color('lightcoral')
                if v.get_patch_by_id('11'): v.get_patch_by_id('11').set_color('mediumpurple')

                if v.get_patch_by_id('10'): v.get_patch_by_id('10').set_alpha(0.6)
                if v.get_patch_by_id('01'): v.get_patch_by_id('01').set_alpha(0.6)
                if v.get_patch_by_id('11'): v.get_patch_by_id('11').set_alpha(0.7)

                # Set label text sizes if needed
                # for text in v.set_labels:
                #     if text: text.set_fontsize(12)
                # for text in v.subset_labels:
                #     if text: text.set_fontsize(10)
            except AttributeError as e:
                 print(f"Warning: Could not fully customize Venn diagram: {e}")


        plt.title('Overlap in Suspicious User Detection')
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "suspicious_users_overlap.png")
        plt.savefig(plot_path)
        print(f"Venn diagram saved to {plot_path}")
        plt.close()
    else:
        print("matplotlib_venn not installed. Skipping Venn diagram generation.")

    # Create a table of users detected as suspicious by only one model
    # Merge based on user_id to get data for discrepancy analysis
    merged_suspicious = lda_suspicious.merge(
        bert_suspicious, on='user_id', suffixes=('_lda', '_bert'), how='outer'
    )

    discrepancy_data = []

    # Iterate through merged data to find discrepancies
    for _, row in merged_suspicious.iterrows():
        user_id = row['user_id']
        is_suspicious_lda = row.get('suspicious_lda', False)
        is_suspicious_bert = row.get('suspicious_bert', False)

        # Use .get() with defaults for potentially missing columns after outer merge
        if is_suspicious_lda and not is_suspicious_bert:
            discrepancy_data.append({
                'user_id': user_id,
                'model': 'LDA only',
                'lda_gini': row.get('gini_coefficient_lda', np.nan),
                'bert_gini': row.get('gini_coefficient_bert', np.nan),
                'post_count': row.get('post_count_lda', row.get('post_count_bert', np.nan)), # Get count from either
                'lda_duplicate_ratio': row.get('duplicate_post_ratio_lda', np.nan),
                'bert_duplicate_ratio': row.get('duplicate_post_ratio_bert', np.nan)
            })
        elif not is_suspicious_lda and is_suspicious_bert:
             discrepancy_data.append({
                'user_id': user_id,
                'model': 'BERTopic only',
                'lda_gini': row.get('gini_coefficient_lda', np.nan),
                'bert_gini': row.get('gini_coefficient_bert', np.nan),
                'post_count': row.get('post_count_bert', row.get('post_count_lda', np.nan)), # Get count from either
                'lda_duplicate_ratio': row.get('duplicate_post_ratio_lda', np.nan),
                'bert_duplicate_ratio': row.get('duplicate_post_ratio_bert', np.nan)
            })


    # Create DataFrame and save to CSV
    if discrepancy_data:
        discrepancy_df = pd.DataFrame(discrepancy_data)
        csv_path = os.path.join(output_dir, "suspicious_user_discrepancies.csv")
        discrepancy_df.to_csv(csv_path, index=False)
        print(f"Suspicious user discrepancy data saved to {csv_path}")
    else:
        print("No discrepancies found in suspicious user detection (or data was missing).")

    return summary


def topic_coherence_comparison(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Compare topic coherence between LDA and BERTopic.
    Placeholder: Actual coherence calculation depends on how it's done in the base classes.
    Assume base classes save a 'topic_coherence.json' if calculated.

    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs
    """
    # Try to load coherence scores if available
    lda_coherence_path = os.path.join(lda_dir, "topic_coherence.json") # Assuming this file is saved by LDA system
    bert_coherence_path = os.path.join(bert_dir, "topic_coherence.json")# Assuming this file is saved by BERTopic system

    lda_coherence = None
    bert_coherence = None

    try:
        if os.path.exists(lda_coherence_path):
            with open(lda_coherence_path, 'r') as f:
                lda_coherence = json.load(f)
        else:
            print(f"LDA coherence file not found: {lda_coherence_path}")
    except Exception as e:
        print(f"Error loading LDA coherence: {e}")


    try:
        if os.path.exists(bert_coherence_path):
            with open(bert_coherence_path, 'r') as f:
                bert_coherence = json.load(f)
        else:
            print(f"BERTopic coherence file not found: {bert_coherence_path}")
    except Exception as e:
        print(f"Error loading BERTopic coherence: {e}")


    if not lda_coherence and not bert_coherence:
        print("Coherence scores not available for comparison for either model.")
        return None

    # Extract mean scores, defaulting to NaN if not found
    # Adjust the key ('coherence_mean') if your base classes use a different key
    lda_coherence_mean = lda_coherence.get('coherence_mean', np.nan) if lda_coherence else np.nan
    bert_coherence_mean = bert_coherence.get('coherence_mean', np.nan) if bert_coherence else np.nan

    # Only proceed if at least one score is valid
    if np.isnan(lda_coherence_mean) and np.isnan(bert_coherence_mean):
        print("No valid coherence scores found in loaded files.")
        return None

    # Create bar chart
    plt.figure(figsize=(8, 6))
    models = []
    coherence_scores = []
    colors = []

    if not np.isnan(lda_coherence_mean):
        models.append('LDA')
        coherence_scores.append(lda_coherence_mean)
        colors.append('skyblue')
    if not np.isnan(bert_coherence_mean):
        models.append('BERTopic')
        coherence_scores.append(bert_coherence_mean)
        colors.append('lightcoral')


    plt.bar(models, coherence_scores, color=colors, alpha=0.7)
    plt.title('Topic Coherence Comparison')
    plt.ylabel('Mean Coherence Score (Higher is Better)')
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, v in enumerate(coherence_scores):
        plt.text(i, v + 0.01 * np.nanmax(coherence_scores), f"{v:.3f}", ha='center', va='bottom') # Adjust offset

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "coherence_comparison.png")
    plt.savefig(plot_path)
    print(f"Coherence comparison plot saved to {plot_path}")
    plt.close()

    # Save summary
    summary = {
        'lda_coherence_mean': lda_coherence_mean,
        'bert_coherence_mean': bert_coherence_mean
    }
    if not np.isnan(lda_coherence_mean) and not np.isnan(bert_coherence_mean):
         summary['difference'] = bert_coherence_mean - lda_coherence_mean
         if lda_coherence_mean != 0: # Avoid division by zero
             summary['percent_improvement_over_lda'] = (summary['difference'] / abs(lda_coherence_mean)) * 100
         else:
              summary['percent_improvement_over_lda'] = np.inf if summary['difference'] > 0 else (-np.inf if summary['difference'] < 0 else 0)


    summary_path = os.path.join(output_dir, "coherence_comparison.json")
    try:
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=handle_numpy_types) # Use handler
        print(f"Coherence comparison summary saved to {summary_path}")
    except Exception as e:
        print(f"Error saving coherence comparison summary JSON: {e}")


    return summary


def topic_words_comparison(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Compare top words for topics between LDA and BERTopic.
    This is more qualitative and will create a side-by-side comparison.

    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs
    """
    # Load topic words
    lda_topic_words_path = os.path.join(lda_dir, "topic_words.csv")
    bert_topic_words_path = os.path.join(bert_dir, "topic_words.csv")

    lda_words = None
    bert_words = None

    if os.path.exists(lda_topic_words_path):
        try:
            lda_words = pd.read_csv(lda_topic_words_path)
        except Exception as e:
            print(f"Error loading LDA topic words: {e}")
    else:
        print(f"LDA topic words file not found: {lda_topic_words_path}")


    if os.path.exists(bert_topic_words_path):
         try:
            bert_words = pd.read_csv(bert_topic_words_path)
            # BERTopic often includes topic -1 (outliers), filter it out for comparison
            if 'topic_id' in bert_words.columns:
                 bert_words = bert_words[bert_words['topic_id'] != -1].copy()
         except Exception as e:
            print(f"Error loading BERTopic topic words: {e}")
    else:
        print(f"BERTopic topic words file not found: {bert_topic_words_path}")


    if lda_words is None and bert_words is None:
        print("Topic word data not available for comparison for either model.")
        return None

    # --- HTML Report Generation ---
    html = "<html><head><title>Topic Comparison</title><style>"
    html += "body { font-family: sans-serif; margin: 20px; } "
    html += "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; } "
    html += "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; vertical-align: top; } "
    html += "th { background-color: #f2f2f2; } "
    html += ".lda { background-color: #e6f3ff; } "
    html += ".bert { background-color: #ffe6e6; } "
    html += "h1, h2 { color: #333; } "
    html += "</style></head><body>"
    html += "<h1>Topic Words Comparison: LDA vs BERTopic</h1>"

    # Only proceed with parts where data exists
    if lda_words is not None and bert_words is not None:
        # Create a side-by-side comparison of top topics
        # Determine number of topics to show based on available data
        num_lda_topics = lda_words['topic_id'].nunique() if lda_words is not None else 0
        num_bert_topics = bert_words['topic_id'].nunique() if bert_words is not None else 0
        num_topics_to_compare = min(5, num_lda_topics, num_bert_topics) if (num_lda_topics > 0 and num_bert_topics > 0) else 0


        if num_topics_to_compare > 0:
            html += "<h2>Top Topics Comparison (Side-by-Side)</h2>"
            html += "<p>Displaying the top words for the first few topics discovered by each model.</p>"
            html += "<table><thead><tr><th>LDA Topic ID</th><th>Top 10 Words (LDA)</th><th>BERTopic Topic ID</th><th>Top 10 Words (BERTopic)</th></tr></thead><tbody>"

            # Get lists of unique topic IDs
            lda_topic_ids = sorted(lda_words['topic_id'].unique())
            bert_topic_ids = sorted(bert_words['topic_id'].unique())


            for i in range(num_topics_to_compare):
                lda_topic = lda_topic_ids[i]
                bert_topic = bert_topic_ids[i]

                lda_top_words_list = lda_words[lda_words['topic_id'] == lda_topic].sort_values('rank').head(10)['word'].tolist()
                bert_top_words_list = bert_words[bert_words['topic_id'] == bert_topic].sort_values('rank').head(10)['word'].tolist()

                html += f"<tr>"
                html += f"<td class='lda'>{lda_topic}</td>"
                html += f"<td class='lda'>{', '.join(lda_top_words_list)}</td>"
                html += f"<td class='bert'>{bert_topic}</td>"
                html += f"<td class='bert'>{', '.join(bert_top_words_list)}</td>"
                html += f"</tr>"

            html += "</tbody></table>"
        else:
             html += "<p>Cannot perform side-by-side comparison due to missing topic data for one or both models.</p>"

        # Attempt to find similar topics between models based on word overlap
        html += "<h2>Topic Similarity Analysis (Based on Word Overlap)</h2>"
        html += "<p>This table shows pairs of topics (one from LDA, one from BERTopic) that share a significant number of top words (Jaccard Similarity > 0.1).</p>"

        # Create sets of words for each topic
        lda_topic_word_sets = {}
        bert_topic_word_sets = {}

        if lda_words is not None:
            for topic_id in lda_words['topic_id'].unique():
                 # Consider top N words for similarity calculation (e.g., top 20)
                 words = set(lda_words[lda_words['topic_id'] == topic_id].sort_values('rank').head(20)['word'].tolist())
                 if words: lda_topic_word_sets[topic_id] = words

        if bert_words is not None:
            for topic_id in bert_words['topic_id'].unique():
                 words = set(bert_words[bert_words['topic_id'] == topic_id].sort_values('rank').head(20)['word'].tolist())
                 if words: bert_topic_word_sets[topic_id] = words

        topic_similarities = []
        if lda_topic_word_sets and bert_topic_word_sets: # Only calculate if both have data
            for lda_topic, lda_words_set in lda_topic_word_sets.items():
                for bert_topic, bert_words_set in bert_topic_word_sets.items():
                    intersection = len(lda_words_set.intersection(bert_words_set))
                    union = len(lda_words_set.union(bert_words_set))

                    if union > 0:
                        similarity = intersection / union
                        # Only consider similarities above a threshold
                        if similarity > 0.1: # Similarity threshold
                            shared_words_list = sorted(list(lda_words_set.intersection(bert_words_set)))
                            topic_similarities.append({
                                'lda_topic': lda_topic,        # Will be handled by default handler
                                'bert_topic': bert_topic,      # Will be handled by default handler
                                'similarity': similarity,    # Will be handled by default handler
                                'shared_words': shared_words_list,
                                'num_shared': len(shared_words_list)
                            })

            # Sort by similarity
            topic_similarities.sort(key=lambda x: x['similarity'], reverse=True)

            # Create table of similar topics
            if topic_similarities:
                html += "<table><thead><tr><th>LDA Topic ID</th><th>BERTopic Topic ID</th><th>Jaccard Similarity</th><th>Number Shared</th><th>Shared Words (Top 10)</th></tr></thead><tbody>"
                for sim in topic_similarities[:15]:  # Show top 15 similar pairs
                    html += f"<tr>"
                    html += f"<td class='lda'>{sim['lda_topic']}</td>"
                    html += f"<td class='bert'>{sim['bert_topic']}</td>"
                    html += f"<td>{sim['similarity']:.3f}</td>"
                    html += f"<td>{sim['num_shared']}</td>"
                    html += f"<td>{', '.join(sim['shared_words'][:10])}</td>" # Show only first 10 shared
                    html += f"</tr>"
                html += "</tbody></table>"
            else:
                html += "<p>No topic pairs found with Jaccard similarity > 0.1 based on top 20 words.</p>"
        else:
            html += "<p>Could not calculate topic similarities due to missing topic word data for one or both models.</p>"


        # Save similarity data to JSON using the default handler
        json_path = os.path.join(output_dir, "topic_similarities.json")
        print(f"Saving topic similarities to {json_path}")
        try:
            with open(json_path, 'w') as f:
                json.dump(topic_similarities, f, indent=2, default=handle_numpy_types)
            print("Topic similarities saved successfully.")
        except Exception as e:
            print(f"Error saving topic similarities JSON: {e}")
            # Optionally save raw data if JSON fails
            # try:
            #     with open(os.path.join(output_dir, "topic_similarities_raw.pkl"), 'wb') as f_pkl:
            #         pickle.dump(topic_similarities, f_pkl)
            #     print("Saved raw similarities to PKL due to JSON error.")
            # except Exception as pkl_e:
            #      print(f"Could not save raw similarities either: {pkl_e}")


    else:
         html += "<p>Could not perform detailed topic word comparison because data was missing for at least one model.</p>"
         topic_similarities = None # Ensure it's defined


    html += "</body></html>"

    # Save HTML file
    html_path = os.path.join(output_dir, "topic_words_comparison.html")
    try:
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Topic words comparison report saved to {html_path}")
    except Exception as e:
        print(f"Error saving topic words comparison HTML: {e}")

    return topic_similarities


def generate_comparison_report(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Generate a comprehensive comparison report in HTML format.

    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs (report will be saved here)
    """
    # Load metrics and summaries, handle potential errors
    def load_json_safe(file_path):
        if not os.path.exists(file_path):
            print(f"Report generation: File not found {file_path}")
            return {}
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"Report generation: Error decoding JSON from {file_path}")
            return {}
        except Exception as e:
            print(f"Report generation: Error loading {file_path}: {e}")
            return {}

    narrowness_summary = load_json_safe(os.path.join(output_dir, "narrowness_metrics_summary.json"))
    suspicious_summary = load_json_safe(os.path.join(output_dir, "suspicious_detection_summary.json"))
    coherence_summary = load_json_safe(os.path.join(output_dir, "coherence_comparison.json"))
    topic_similarities = load_json_safe(os.path.join(output_dir, "topic_similarities.json")) # Assumes this was saved

    # Load model configs
    lda_config = load_json_safe(os.path.join(lda_dir, "config.json"))
    bert_config = load_json_safe(os.path.join(bert_dir, "config.json"))


    # --- Create HTML Report ---
    html = "<html><head><title>Model Comparison Report</title><style>"
    html += "body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; } "
    html += "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); } "
    html += "th, td { padding: 10px 12px; text-align: left; border: 1px solid #ddd; } "
    html += "th { background-color: #f2f2f2; font-weight: bold; } "
    html += "tr:nth-child(even) { background-color: #f9f9f9; } "
    html += ".lda { background-color: #e6f3ff; } "
    html += ".bert { background-color: #ffe6e6; } "
    html += "h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; } "
    html += "h2 { color: #34495e; border-bottom: 1px solid #bdc3c7; padding-bottom: 5px; margin-top: 30px; } "
    html += "h3 { color: #7f8c8d; margin-top: 20px; } "
    html += "img { max-width: 80%; height: auto; display: block; margin: 15px auto; border: 1px solid #ddd; padding: 5px; box-shadow: 0 2px 3px rgba(0,0,0,0.1); } "
    html += ".section { margin-bottom: 30px; background-color: #fff; padding: 20px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); } "
    html += "ul { padding-left: 20px; } li { margin-bottom: 8px; } "
    html += "a { color: #3498db; text-decoration: none; } a:hover { text-decoration: underline; } "
    html += ".code { background-color: #ecf0f1; padding: 2px 5px; border-radius: 3px; font-family: monospace; } "
    html += ".value-positive { color: #27ae60; } .value-negative { color: #c0392b; } .value-neutral { color: #7f8c8d; }"
    html += "</style></head><body>"

    html += "<h1>Model Comparison Report: LDA vs BERTopic</h1>"

    # Introduction
    html += "<div class='section'>"
    html += "<h2>Introduction</h2>"
    html += "<p>This report compares the performance and results of two topic modeling approaches applied to the same dataset:</p>"
    html += "<ul>"
    html += "<li><strong>LDA (Latent Dirichlet Allocation):</strong> A traditional probabilistic topic modeling method focused on word co-occurrences.</li>"
    html += "<li><strong>BERTopic:</strong> A modern approach leveraging transformer-based sentence embeddings and clustering techniques.</li>"
    html += "</ul>"
    html += "<p>The comparison focuses on user topic distributions, suspicious user detection, topic coherence (if available), and qualitative topic assessment.</p>"
    html += "</div>"

    # Model Configuration
    html += "<div class='section'>"
    html += "<h2>Model Configuration</h2>"
    html += "<table>"
    html += "<thead><tr><th>Parameter</th><th class='lda'>LDA Config</th><th class='bert'>BERTopic Config</th></tr></thead><tbody>"

    # Dynamically add key parameters found in configs
    all_keys = set(lda_config.keys()) | set(bert_config.keys())
    param_map = { # Optional: map keys to nicer names
        'num_topics': 'Number of Topics', 'min_topic_size': 'Min Topic Size',
        'combine_by_window': 'Combine Posts by Window', 'lemmatize': 'Lemmatization (LDA)',
        'language_model': 'Language Model (BERTopic)', 'n_neighbors': 'UMAP Neighbors (BERTopic)',
        'n_components': 'UMAP Components (BERTopic)', 'min_cluster_size': 'HDBSCAN Min Cluster Size (BERTopic)'
    }

    for key in sorted(list(all_keys)):
         display_key = param_map.get(key, key.replace('_', ' ').title())
         lda_val = lda_config.get(key, 'N/A')
         bert_val = bert_config.get(key, 'N/A')
         html += f"<tr><td>{display_key}</td><td class='lda'>{lda_val}</td><td class='bert'>{bert_val}</td></tr>"

    html += "</tbody></table>"
    html += "</div>"

    # Performance Comparison Section
    html += "<div class='section'>"
    html += "<h2>Performance & Results Comparison</h2>"

    # Coherence scores
    if coherence_summary and ('lda_coherence_mean' in coherence_summary or 'bert_coherence_mean' in coherence_summary) \
       and not (np.isnan(coherence_summary.get('lda_coherence_mean', np.nan)) and np.isnan(coherence_summary.get('bert_coherence_mean', np.nan))):
        html += "<h3>Topic Coherence</h3>"
        html += "<p>Topic coherence measures the semantic interpretability of topics based on their top words. Higher scores are generally better. <i>(Note: Coherence calculation methods might differ between models.)</i></p>"
        html += "<table><thead><tr><th>Model</th><th>Mean Coherence Score</th>"
        if 'difference' in coherence_summary: html += "<th>Difference (BERTopic - LDA)</th><th>% Improvement vs LDA</th>"
        html += "</tr></thead><tbody>"

        lda_score = coherence_summary.get('lda_coherence_mean', np.nan)
        bert_score = coherence_summary.get('bert_coherence_mean', np.nan)

        html += f"<tr><td class='lda'>LDA</td><td>{'N/A' if np.isnan(lda_score) else f'{lda_score:.4f}'}</td>"
        if 'difference' in coherence_summary: html += f"<td rowspan='2'>{coherence_summary['difference']:.4f}</td><td rowspan='2'>{coherence_summary.get('percent_improvement_over_lda', 0):.2f}%</td>"
        html += "</tr>"
        html += f"<tr><td class='bert'>BERTopic</td><td>{'N/A' if np.isnan(bert_score) else f'{bert_score:.4f}'}</td></tr>"



        html += "</tbody></table>"
        # Embed the coherence plot if it exists
        coherence_plot_path = "coherence_comparison.png"
        if os.path.exists(os.path.join(output_dir, coherence_plot_path)):
             html += f"<img src='{coherence_plot_path}' alt='Coherence Comparison Chart'>"
        else:
             html += "<p><i>Coherence comparison plot not found.</i></p>"
    elif coherence_summary: # If summary dict exists but has no valid data
         html += "<h3>Topic Coherence</h3><p>No valid coherence scores were found or calculated for comparison.</p>"


    # User narrowness metrics
    if narrowness_summary:
        html += "<h3>User Topic Narrowness Correlation</h3>"
        html += "<p>This section compares how similarly the two models assess user topic concentration (narrowness/diversity) using various metrics. High correlation suggests agreement between models.</p>"
        html += "<table><thead>"
        html += "<tr><th>Metric</th><th>Correlation (LDA vs BERTopic)</th><th>LDA Mean ± Std Dev</th><th>BERTopic Mean ± Std Dev</th><th>Mean Difference (BERTopic - LDA)</th></tr>"
        html += "</thead><tbody>"

        for metric, data in narrowness_summary.items():
            metric_title = metric.replace('_', ' ').title()
            corr = data.get('correlation', np.nan)
            lda_m, lda_s = data.get('lda_mean', np.nan), data.get('lda_std', np.nan)
            bert_m, bert_s = data.get('bert_mean', np.nan), data.get('bert_std', np.nan)
            diff = data.get('mean_difference', np.nan)

            # Format values safely
            corr_str = f"{corr:.3f}" if not np.isnan(corr) else "N/A"
            lda_str = f"{lda_m:.3f} ± {lda_s:.3f}" if not np.isnan(lda_m) and not np.isnan(lda_s) else "N/A"
            bert_str = f"{bert_m:.3f} ± {bert_s:.3f}" if not np.isnan(bert_m) and not np.isnan(bert_s) else "N/A"
            diff_str = f"{diff:+.3f}" if not np.isnan(diff) else "N/A" # Add sign to difference

            html += f"<tr><td>{metric_title}</td>"
            html += f"<td>{corr_str}</td>"
            html += f"<td class='lda'>{lda_str}</td>"
            html += f"<td class='bert'>{bert_str}</td>"
            html += f"<td>{diff_str}</td></tr>"

        html += "</tbody></table>"
        # Embed the narrowness plot if it exists
        narrowness_plot_path = "narrowness_metrics_comparison.png"
        if os.path.exists(os.path.join(output_dir, narrowness_plot_path)):
            html += f"<img src='{narrowness_plot_path}' alt='Narrowness Metrics Comparison Scatter Plots'>"
        else:
            html += "<p><i>Narrowness metrics comparison plot not found.</i></p>"
    else:
         html += "<h3>User Topic Narrowness Correlation</h3><p>Narrowness metrics summary data not found.</p>"

    # Suspicious user detection
    if suspicious_summary:
        html += "<h3>Suspicious User Detection</h3>"
        html += "<p>This compares the sets of users flagged as potentially anomalous (e.g., bots, spammers) by each model based on posting frequency, topic narrowness, and content duplication.</p>"
        html += "<table><thead>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        html += "</thead><tbody>"

        # Define metrics and format them
        s_metrics = [
            ('Total Suspicious Users (LDA)', suspicious_summary.get('total_suspicious_lda', 'N/A')),
            ('Total Suspicious Users (BERTopic)', suspicious_summary.get('total_suspicious_bert', 'N/A')),
            ('Users Flagged by Both (Overlap)', suspicious_summary.get('overlap_count', 'N/A')),
            ('Overlap % of LDA Total', f"{suspicious_summary.get('overlap_percent_of_lda', np.nan):.1f}%" if not np.isnan(suspicious_summary.get('overlap_percent_of_lda', np.nan)) else "N/A"),
            ('Overlap % of BERTopic Total', f"{suspicious_summary.get('overlap_percent_of_bert', np.nan):.1f}%" if not np.isnan(suspicious_summary.get('overlap_percent_of_bert', np.nan)) else "N/A"),
            ('Users Flagged by LDA Only', suspicious_summary.get('lda_only_count', 'N/A')),
            ('Users Flagged by BERTopic Only', suspicious_summary.get('bert_only_count', 'N/A'))
        ]

        for label, value in s_metrics:
             html += f"<tr><td>{label}</td><td>{value}</td></tr>"

        html += "</tbody></table>"
        # Embed the Venn diagram plot if it exists
        venn_plot_path = "suspicious_users_overlap.png"
        if os.path.exists(os.path.join(output_dir, venn_plot_path)):
            html += f"<img src='{venn_plot_path}' alt='Suspicious Users Overlap Venn Diagram'>"
        else:
            html += "<p><i>Suspicious users overlap plot (Venn diagram) not found or library not installed.</i></p>"
        # Link to discrepancy CSV if it exists
        discrepancy_csv_path = "suspicious_user_discrepancies.csv"
        if os.path.exists(os.path.join(output_dir, discrepancy_csv_path)):
            html += f"<p>See <a href='{discrepancy_csv_path}'>suspicious_user_discrepancies.csv</a> for users flagged by only one model.</p>"

    else:
         html += "<h3>Suspicious User Detection</h3><p>Suspicious user detection summary data not found.</p>"

    html += "</div>" # End Performance Comparison Section

    # Topic Similarity Analysis
    html += "<div class='section'>"
    html += "<h2>Topic Word & Similarity Analysis</h2>"
    # Link to detailed HTML comparison
    words_html_path = "topic_words_comparison.html"
    if os.path.exists(os.path.join(output_dir, words_html_path)):
         html += f"<p>View the detailed side-by-side topic word lists and similarity analysis: <a href='{words_html_path}'>Topic Words Comparison Report</a></p>"
    else:
         html += "<p>Detailed topic words comparison HTML report not found.</p>"

    if isinstance(topic_similarities, list) and topic_similarities: # Check if it's a non-empty list
        html += "<h3>Most Similar Topic Pairs (Jaccard > 0.1)</h3>"
        html += "<p>Top pairs of topics from LDA and BERTopic based on shared words (calculated using top 20 words per topic).</p>"
        html += "<table><thead>"
        html += "<tr><th>LDA Topic ID</th><th>BERTopic Topic ID</th><th>Similarity Score</th><th>Number Shared Words</th><th>Example Shared Words</th></tr>"
        html += "</thead><tbody>"

        for sim in topic_similarities[:10]:  # Show top 10 similar pairs
            shared_words_str = ", ".join(sim.get('shared_words', [])[:5]) # Show first 5 shared
            html += f"<tr>"
            html += f"<td class='lda'>{sim.get('lda_topic','N/A')}</td>"
            html += f"<td class='bert'>{sim.get('bert_topic','N/A')}</td>"
            html += f"<td>{sim.get('similarity', np.nan):.3f}</td>"
            html += f"<td>{sim.get('num_shared','N/A')}</td>"
            html += f"<td>{shared_words_str}</td>"
            html += f"</tr>"

        html += "</tbody></table>"
    elif isinstance(topic_similarities, dict) and not topic_similarities: # Handle empty dict from failed load
         html += "<p>Topic similarity data could not be loaded.</p>"
    elif not isinstance(topic_similarities, list): # Handle incorrect type
         html += f"<p>Topic similarity data is not in the expected format (found type: {type(topic_similarities)}).</p>"
    else: # Handle empty list
         html += "<p>No significant topic similarity (Jaccard > 0.1) found between LDA and BERTopic topics.</p>"

    html += "</div>" # End Topic Similarity Section


    # Conclusions
    html += "<div class='section'>"
    html += "<h2>Conclusions & Observations</h2>"
    conclusions = []

    # Coherence conclusion
    if coherence_summary and not np.isnan(coherence_summary.get('lda_coherence_mean', np.nan)) and not np.isnan(coherence_summary.get('bert_coherence_mean', np.nan)):
        diff = coherence_summary.get('difference', 0)
        if diff > 0.01: # Threshold for meaningful difference
            conclusions.append("BERTopic produced significantly more coherent topics than LDA according to the calculated metric, suggesting better semantic interpretability with the embedding-based approach for this dataset.")
        elif diff < -0.01:
            conclusions.append("LDA produced significantly more coherent topics than BERTopic, suggesting the traditional probabilistic approach might be better suited for capturing interpretable word patterns in this dataset.")
        else:
            conclusions.append("LDA and BERTopic produced topics with comparable coherence scores.")
    elif coherence_summary:
         conclusions.append("Coherence scores were only available for one model, or were not calculated.")

    # Narrowness metrics conclusion
    if narrowness_summary and 'gini_coefficient' in narrowness_summary:
        corr = narrowness_summary['gini_coefficient'].get('correlation', np.nan)
        if not np.isnan(corr):
            if corr > 0.7:
                conclusions.append("Strong positive correlation found in user topic narrowness (Gini Coefficient) between models, indicating they largely agree on which users have focused vs. diverse posting behavior.")
            elif corr > 0.4:
                conclusions.append("Moderate positive correlation in user topic narrowness suggests some agreement, but also that the models capture slightly different aspects of user focus.")
            elif corr > 0.1:
                 conclusions.append("Weak positive correlation in user topic narrowness; the models seem to provide different perspectives on user behavior concentration.")
            else:
                 conclusions.append("Low or negative correlation in user topic narrowness; the models significantly disagree on user behavior concentration.")
        else:
             conclusions.append("Could not calculate correlation for user narrowness metrics.")

    # Suspicious detection conclusion
    if suspicious_summary:
        overlap_lda = suspicious_summary.get('overlap_percent_of_lda', np.nan)
        overlap_bert = suspicious_summary.get('overlap_percent_of_bert', np.nan)
        if not np.isnan(overlap_lda) and not np.isnan(overlap_bert):
            avg_overlap = (overlap_lda + overlap_bert) / 2
            if avg_overlap > 60:
                conclusions.append("High agreement (% overlap > 60%) between models on suspicious user detection, suggesting a core set of anomalous users are identified consistently.")
            elif avg_overlap > 30:
                conclusions.append("Moderate agreement (% overlap 30-60%) on suspicious users. Combining insights from both models might be beneficial for robust detection.")
            else:
                conclusions.append("Low agreement (% overlap < 30%) on suspicious users. Each model likely flags different types of anomalies (e.g., LDA sensitive to term frequency, BERTopic to semantic repetition).")
        else:
             conclusions.append("Could not determine overlap percentage for suspicious user detection.")


    # Overall conclusion (simple example)
    if coherence_summary and coherence_summary.get('difference', 0) > 0.01:
        conclusions.append("Overall, BERTopic appears to offer advantages in topic quality (coherence) for this specific dataset run, while narrowness agreement varies. BERTopic's embedding approach may capture nuances missed by LDA.")
    elif coherence_summary and coherence_summary.get('difference', 0) < -0.01:
        conclusions.append("Overall, LDA demonstrates stronger performance in topic coherence for this dataset run. Both models provide valuable perspectives, with LDA potentially being more effective for traditional term-based topic interpretation here.")
    else:
        conclusions.append("Overall, both models offer potentially valuable insights. Performance appears comparable in key metrics like coherence (or coherence wasn't comparable). The choice may depend on whether semantic nuance (BERTopic) or term-frequency patterns (LDA) are prioritized.")

    # Add conclusions to HTML
    if conclusions:
        html += "<ul>"
        for conclusion in conclusions:
            html += f"<li>{conclusion}</li>"
        html += "</ul>"
    else:
         html += "<p>No automated conclusions could be drawn due to missing summary data.</p>"

    html += "</div>" # End Conclusions

    # Footer
    html += "<div class='section' style='text-align: center; font-size: 0.9em; color: #7f8c8d;'>"
    html += "<p><em>Report generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</em></p>"
    html += "</div>"

    html += "</body></html>"

    # Save HTML report
    report_path = os.path.join(output_dir, "model_comparison_report.html")
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f"Comparison report generated at {report_path}")
    except Exception as e:
        print(f"Error saving comparison report HTML: {e}")


# --- Main Execution Block ---
# This block is typically removed if comparison_script.py is only meant to be imported
# Keeping it allows running this script directly for comparison if needed.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare LDA and BERTopic model results')
    parser.add_argument('lda_dir', help='Directory containing LDA results (e.g., topic_analysis_results/lda_results)')
    parser.add_argument('bert_dir', help='Directory containing BERTopic results (e.g., topic_analysis_results/bertopic_results)')
    parser.add_argument('output_dir', help='Directory to save comparison outputs (e.g., topic_analysis_results/model_comparison)')

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    compare_model_results(args.lda_dir, args.bert_dir, args.output_dir)

    print("\nDirect comparison script execution finished.")
    print(f"Comparison results saved in: {args.output_dir}")