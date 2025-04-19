import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple
import os
import pickle

from user_topic_modeling import UserTopicModelingSystem
from bertopic_extension import BERTopicModelingSystem


def run_comparison(input_file: str, output_dir: str, 
                  lda_topics: int = 20, bert_topics: int = 20,
                  combine_by_window: bool = True):
    """
    Run both LDA and BERTopic models on the same dataset and compare results.
    
    Args:
        input_file: Path to input data file
        output_dir: Directory to save comparison outputs
        lda_topics: Number of topics for LDA model
        bert_topics: Number of topics for BERTopic model
        combine_by_window: Whether to combine posts into time windows
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create subdirectories for model outputs
    lda_dir = os.path.join(output_dir, "lda_results")
    bert_dir = os.path.join(output_dir, "bertopic_results")
    os.makedirs(lda_dir, exist_ok=True)
    os.makedirs(bert_dir, exist_ok=True)
    
    # Step 1: Run LDA model
    print("Running LDA model...")
    lda_system = UserTopicModelingSystem(
        num_topics=lda_topics,
        time_bin='week',
        lemmatize=True,
        extra_stopwords=['rt', 'http', 'https', 'amp']
    )
    
    lda_results = lda_system.run_full_pipeline(
        input_file=input_file,
        output_dir=lda_dir,
        combine_by_window=combine_by_window,
        visualize_topics=True,
        detect_suspicious=True
    )
    
    # Step 2: Run BERTopic model
    print("\nRunning BERTopic model...")
    bert_system = BERTopicModelingSystem(
        language_model='all-MiniLM-L6-v2',
        time_bin='week',
        min_topic_size=10,
        extra_stopwords=['rt', 'http', 'https', 'amp'],
        n_neighbors=15,
        n_components=5,
        min_cluster_size=15
    )
    
    bert_results = bert_system.run_full_pipeline(
        input_file=input_file,
        output_dir=bert_dir,
        combine_by_window=combine_by_window,
        visualize_topics=True,
        detect_suspicious=True
    )
    
    # Step 3: Compare results
    print("\nComparing results...")
    compare_model_results(lda_dir, bert_dir, output_dir)
    
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
    # Load user topic data
    lda_user_data = pd.read_csv(os.path.join(lda_dir, "user_topic_data.csv"))
    bert_user_data = pd.read_csv(os.path.join(bert_dir, "user_topic_data.csv"))
    
    # Load suspicious users data
    lda_suspicious = pd.read_csv(os.path.join(lda_dir, "suspicious_users.csv"))
    bert_suspicious = pd.read_csv(os.path.join(bert_dir, "suspicious_users.csv"))
    
    # 1. Compare user narrowness metrics
    compare_narrowness_metrics(lda_user_data, bert_user_data, output_dir)
    
    # 2. Compare suspicious user detection
    compare_suspicious_detection(lda_suspicious, bert_suspicious, output_dir)
    
    # 3. Compare topic coherence (if available)
    topic_coherence_comparison(lda_dir, bert_dir, output_dir)
    
    # 4. Compare topic words (qualitative analysis)
    topic_words_comparison(lda_dir, bert_dir, output_dir)
    
    # 5. Generate comparison report
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
    merged_data = lda_data[['user_id', 'gini_coefficient', 'shannon_entropy', 'top1_ratio']].merge(
        bert_data[['user_id', 'gini_coefficient', 'shannon_entropy', 'top1_ratio']],
        on='user_id',
        suffixes=('_lda', '_bert')
    )
    
    # Create correlation plots
    metrics = ['gini_coefficient', 'shannon_entropy', 'top1_ratio']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, metric in enumerate(metrics):
        lda_metric = f"{metric}_lda"
        bert_metric = f"{metric}_bert"
        
        # Calculate correlation
        corr = merged_data[lda_metric].corr(merged_data[bert_metric])
        
        # Create scatter plot
        axes[i].scatter(merged_data[lda_metric], merged_data[bert_metric], alpha=0.5)
        axes[i].set_title(f"{metric.replace('_', ' ').title()} Correlation: {corr:.3f}")
        axes[i].set_xlabel(f"LDA {metric.replace('_', ' ').title()}")
        axes[i].set_ylabel(f"BERTopic {metric.replace('_', ' ').title()}")
        
        # Add diagonal line
        min_val = min(merged_data[lda_metric].min(), merged_data[bert_metric].min())
        max_val = max(merged_data[lda_metric].max(), merged_data[bert_metric].max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add regression line
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            merged_data[lda_metric], merged_data[bert_metric]
        )
        axes[i].plot(
            [min_val, max_val], 
            [intercept + slope * min_val, intercept + slope * max_val], 
            'g-', 
            label=f'Regression (r²={r_value**2:.3f})'
        )
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "narrowness_metrics_comparison.png"))
    plt.close()
    
    # Calculate summary statistics
    summary = {}
    for metric in metrics:
        lda_metric = f"{metric}_lda"
        bert_metric = f"{metric}_bert"
        
        summary[metric] = {
            'correlation': merged_data[lda_metric].corr(merged_data[bert_metric]),
            'lda_mean': merged_data[lda_metric].mean(),
            'bert_mean': merged_data[bert_metric].mean(),
            'lda_std': merged_data[lda_metric].std(),
            'bert_std': merged_data[bert_metric].std(),
            'mean_difference': merged_data[bert_metric].mean() - merged_data[lda_metric].mean()
        }
    
    # Save summary
    with open(os.path.join(output_dir, "narrowness_metrics_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return merged_data


def compare_suspicious_detection(lda_suspicious: pd.DataFrame, bert_suspicious: pd.DataFrame, output_dir: str):
    """
    Compare suspicious user detection between LDA and BERTopic.
    
    Args:
        lda_suspicious: DataFrame with LDA suspicious users
        bert_suspicious: DataFrame with BERTopic suspicious users
        output_dir: Directory to save comparison outputs
    """
    # Get lists of suspicious users
    lda_suspicious_users = set(lda_suspicious[lda_suspicious['suspicious']]['user_id'])
    bert_suspicious_users = set(bert_suspicious[bert_suspicious['suspicious']]['user_id'])
    
    # Calculate overlap
    overlap = lda_suspicious_users.intersection(bert_suspicious_users)
    lda_only = lda_suspicious_users - bert_suspicious_users
    bert_only = bert_suspicious_users - lda_suspicious_users
    
    # Create summary
    summary = {
        'total_suspicious_lda': len(lda_suspicious_users),
        'total_suspicious_bert': len(bert_suspicious_users),
        'overlap_count': len(overlap),
        'overlap_percent_of_lda': len(overlap) / max(1, len(lda_suspicious_users)) * 100,
        'overlap_percent_of_bert': len(overlap) / max(1, len(bert_suspicious_users)) * 100,
        'lda_only_count': len(lda_only),
        'bert_only_count': len(bert_only)
    }
    
    # Save summary
    with open(os.path.join(output_dir, "suspicious_detection_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Create Venn diagram
    plt.figure(figsize=(10, 8))
    from matplotlib_venn import venn2
    
    v = venn2([lda_suspicious_users, bert_suspicious_users], 
             set_labels=('LDA Suspicious', 'BERTopic Suspicious'))
    
    # Set colors
    v.get_patch_by_id('10').set_color('blue')
    v.get_patch_by_id('01').set_color('red')
    v.get_patch_by_id('11').set_color('purple')
    
    # Set labels
    v.get_patch_by_id('10').set_alpha(0.5)
    v.get_patch_by_id('01').set_alpha(0.5)
    v.get_patch_by_id('11').set_alpha(0.5)
    
    plt.title('Overlap in Suspicious User Detection')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "suspicious_users_overlap.png"))
    plt.close()
    
    # Create a table of users detected as suspicious by only one model
    discrepancy_data = []
    
    # Add LDA-only suspicious users
    for user_id in lda_only:
        lda_row = lda_suspicious[lda_suspicious['user_id'] == user_id].iloc[0]
        bert_row = bert_suspicious[bert_suspicious['user_id'] == user_id].iloc[0]
        
        discrepancy_data.append({
            'user_id': user_id,
            'model': 'LDA only',
            'lda_gini': lda_row['gini_coefficient'],
            'bert_gini': bert_row['gini_coefficient'],
            'lda_post_count': lda_row['post_count'],
            'bert_post_count': bert_row['post_count'],
            'lda_duplicate_ratio': lda_row['duplicate_post_ratio'],
            'bert_duplicate_ratio': bert_row['duplicate_post_ratio']
        })
    
    # Add BERTopic-only suspicious users
    for user_id in bert_only:
        lda_row = lda_suspicious[lda_suspicious['user_id'] == user_id].iloc[0]
        bert_row = bert_suspicious[bert_suspicious['user_id'] == user_id].iloc[0]
        
        discrepancy_data.append({
            'user_id': user_id,
            'model': 'BERTopic only',
            'lda_gini': lda_row['gini_coefficient'],
            'bert_gini': bert_row['gini_coefficient'],
            'lda_post_count': lda_row['post_count'],
            'bert_post_count': bert_row['post_count'],
            'lda_duplicate_ratio': lda_row['duplicate_post_ratio'],
            'bert_duplicate_ratio': bert_row['duplicate_post_ratio']
        })
    
    # Create DataFrame and save to CSV
    discrepancy_df = pd.DataFrame(discrepancy_data)
    discrepancy_df.to_csv(os.path.join(output_dir, "suspicious_user_discrepancies.csv"), index=False)
    
    return summary


def topic_coherence_comparison(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Compare topic coherence between LDA and BERTopic.
    
    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs
    """
    # Try to load coherence scores if available
    lda_coherence_path = os.path.join(lda_dir, "topic_coherence.json")
    bert_coherence_path = os.path.join(bert_dir, "topic_coherence.json")
    
    lda_coherence = None
    bert_coherence = None
    
    if os.path.exists(lda_coherence_path):
        with open(lda_coherence_path, 'r') as f:
            lda_coherence = json.load(f)
    
    if os.path.exists(bert_coherence_path):
        with open(bert_coherence_path, 'r') as f:
            bert_coherence = json.load(f)
    
    if not lda_coherence or not bert_coherence:
        print("Coherence scores not available for comparison")
        return
    
    # Compare coherence scores
    lda_coherence_mean = lda_coherence.get('coherence_mean', 0)
    bert_coherence_mean = bert_coherence.get('coherence_mean', 0)
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    models = ['LDA', 'BERTopic']
    coherence_scores = [lda_coherence_mean, bert_coherence_mean]
    
    plt.bar(models, coherence_scores, color=['blue', 'red'], alpha=0.7)
    plt.title('Topic Coherence Comparison')
    plt.ylabel('Mean Coherence Score')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(coherence_scores):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "coherence_comparison.png"))
    plt.close()
    
    # Save summary
    summary = {
        'lda_coherence_mean': lda_coherence_mean,
        'bert_coherence_mean': bert_coherence_mean,
        'difference': bert_coherence_mean - lda_coherence_mean,
        'percent_improvement': (bert_coherence_mean - lda_coherence_mean) / lda_coherence_mean * 100 if lda_coherence_mean > 0 else 0
    }
    
    with open(os.path.join(output_dir, "coherence_comparison.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
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
    
    if not os.path.exists(lda_topic_words_path) or not os.path.exists(bert_topic_words_path):
        print("Topic word data not available for comparison")
        return
    
    lda_words = pd.read_csv(lda_topic_words_path)
    bert_words = pd.read_csv(bert_topic_words_path)
    
    # Create a side-by-side comparison of top topics
    num_topics_to_compare = min(5, lda_words['topic_id'].nunique(), bert_words['topic_id'].nunique())
    
    # Get top topics for each model (based on count in topic_info or frequency)
    lda_top_topics = lda_words['topic_id'].value_counts().nlargest(num_topics_to_compare).index.tolist()
    bert_top_topics = bert_words['topic_id'].value_counts().nlargest(num_topics_to_compare).index.tolist()
    
    # Create HTML table for comparison
    html = "<html><head><style>"
    html += "table { border-collapse: collapse; width: 100%; }"
    html += "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }"
    html += "th { background-color: #f2f2f2; }"
    html += ".lda { background-color: #e6f3ff; }"
    html += ".bert { background-color: #ffe6e6; }"
    html += "</style></head><body>"
    
    html += "<h1>Topic Words Comparison: LDA vs BERTopic</h1>"
    
    html += "<h2>Top Topics Comparison</h2>"
    html += "<table><tr><th>LDA Topic</th><th>Top Words (LDA)</th><th>BERTopic Topic</th><th>Top Words (BERTopic)</th></tr>"
    
    for i in range(num_topics_to_compare):
        lda_topic = lda_top_topics[i]
        bert_topic = bert_top_topics[i]
        
        lda_top_words = lda_words[lda_words['topic_id'] == lda_topic].sort_values('rank').head(10)['word'].tolist()
        bert_top_words = bert_words[bert_words['topic_id'] == bert_topic].sort_values('rank').head(10)['word'].tolist()
        
        html += f"<tr>"
        html += f"<td class='lda'>Topic {lda_topic}</td>"
        html += f"<td class='lda'>{', '.join(lda_top_words)}</td>"
        html += f"<td class='bert'>Topic {bert_topic}</td>"
        html += f"<td class='bert'>{', '.join(bert_top_words)}</td>"
        html += f"</tr>"
    
    html += "</table>"
    
    # Attempt to find similar topics between models based on word overlap
    html += "<h2>Topic Similarity Analysis</h2>"
    html += "<p>The following table shows topics from both models that share similar words.</p>"
    
    # Create sets of words for each topic
    lda_topic_word_sets = {}
    bert_topic_word_sets = {}
    
    for topic in lda_words['topic_id'].unique():
        words = lda_words[lda_words['topic_id'] == topic]['word'].tolist()
        lda_topic_word_sets[topic] = set(words)
    
    for topic in bert_words['topic_id'].unique():
        words = bert_words[bert_words['topic_id'] == topic]['word'].tolist()
        bert_topic_word_sets[topic] = set(words)
    
    # Calculate Jaccard similarity between topics
    topic_similarities = []
    
    for lda_topic, lda_words_set in lda_topic_word_sets.items():
        for bert_topic, bert_words_set in bert_topic_word_sets.items():
            intersection = len(lda_words_set.intersection(bert_words_set))
            union = len(lda_words_set.union(bert_words_set))
            
            if union > 0:
                similarity = intersection / union
                
                # Only consider similarities above a threshold
                if similarity > 0.1:
                    topic_similarities.append({
                        'lda_topic': lda_topic,
                        'bert_topic': bert_topic,
                        'similarity': similarity,
                        'shared_words': list(lda_words_set.intersection(bert_words_set))
                    })
    
    # Sort by similarity
    topic_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    # Create table of similar topics
    html += "<table><tr><th>LDA Topic</th><th>BERTopic Topic</th><th>Similarity</th><th>Shared Words</th></tr>"
    
    for sim in topic_similarities[:10]:  # Show top 10 similar pairs
        html += f"<tr>"
        html += f"<td class='lda'>Topic {sim['lda_topic']}</td>"
        html += f"<td class='bert'>Topic {sim['bert_topic']}</td>"
        html += f"<td>{sim['similarity']:.3f}</td>"
        html += f"<td>{', '.join(sim['shared_words'][:10])}</td>"
        html += f"</tr>"
    
    html += "</table>"
    html += "</body></html>"
    
    # Save HTML file
    with open(os.path.join(output_dir, "topic_words_comparison.html"), 'w') as f:
        f.write(html)
    
    # Save similarity data
    with open(os.path.join(output_dir, "topic_similarities.json"), 'w') as f:
        json.dump(topic_similarities, f, indent=2)
    
    return topic_similarities


def generate_comparison_report(lda_dir: str, bert_dir: str, output_dir: str):
    """
    Generate a comprehensive comparison report.
    
    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison outputs
    """
    # Load metrics and summaries
    try:
        with open(os.path.join(output_dir, "narrowness_metrics_summary.json"), 'r') as f:
            narrowness_summary = json.load(f)
    except:
        narrowness_summary = {}
    
    try:
        with open(os.path.join(output_dir, "suspicious_detection_summary.json"), 'r') as f:
            suspicious_summary = json.load(f)
    except:
        suspicious_summary = {}
    
    try:
        with open(os.path.join(output_dir, "coherence_comparison.json"), 'r') as f:
            coherence_summary = json.load(f)
    except:
        coherence_summary = {}
    
    try:
        with open(os.path.join(output_dir, "topic_similarities.json"), 'r') as f:
            topic_similarities = json.load(f)
    except:
        topic_similarities = []
    
    # Create HTML report
    html = "<html><head><style>"
    html += "body { font-family: Arial, sans-serif; margin: 20px; }"
    html += "table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }"
    html += "th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }"
    html += "th { background-color: #f2f2f2; }"
    html += ".lda { background-color: #e6f3ff; }"
    html += ".bert { background-color: #ffe6e6; }"
    html += "h1 { color: #333366; }"
    html += "h2 { color: #666699; border-bottom: 1px solid #cccccc; padding-bottom: 5px; }"
    html += "img { max-width: 100%; }"
    html += ".section { margin-bottom: 30px; }"
    html += "</style></head><body>"
    
    html += "<h1>Model Comparison Report: LDA vs BERTopic</h1>"
    
    # Introduction
    html += "<div class='section'>"
    html += "<h2>Introduction</h2>"
    html += "<p>This report compares the performance of two topic modeling approaches:</p>"
    html += "<ul>"
    html += "<li><strong>LDA (Latent Dirichlet Allocation)</strong>: A traditional probabilistic topic modeling approach.</li>"
    html += "<li><strong>BERTopic</strong>: A modern approach that leverages BERT embeddings for topic modeling.</li>"
    html += "</ul>"
    html += "<p>The models were run on the same dataset with similar parameters.</p>"
    html += "</div>"
    
    # Model Configuration
    html += "<div class='section'>"
    html += "<h2>Model Configuration</h2>"
    html += "<table>"
    html += "<tr><th>Parameter</th><th class='lda'>LDA</th><th class='bert'>BERTopic</th></tr>"
    
    # Load basic configs if available
    try:
        with open(os.path.join(lda_dir, "config.json"), 'r') as f:
            lda_config = json.load(f)
    except:
        lda_config = {}
    
    try:
        with open(os.path.join(bert_dir, "config.json"), 'r') as f:
            bert_config = json.load(f)
    except:
        bert_config = {}
    
    # Add key parameters
    params = [
        ('Number of Topics', 
         lda_config.get('num_topics', 'Not specified'), 
         bert_config.get('num_topics', 'Not specified')),
        ('Preprocessing', 
         lda_config.get('lemmatize', True) and 'Lemmatization' or 'Stemming', 
         'BERT Embeddings'),
        ('Document Combination', 
         lda_config.get('combine_by_window', True) and 'By Time Window' or 'Individual Posts', 
         bert_config.get('combine_by_window', True) and 'By Time Window' or 'Individual Posts')
    ]
    
    for param, lda_value, bert_value in params:
        html += f"<tr><td>{param}</td><td class='lda'>{lda_value}</td><td class='bert'>{bert_value}</td></tr>"
    
    html += "</table>"
    html += "</div>"
    
    # Performance Metrics
    html += "<div class='section'>"
    html += "<h2>Performance Comparison</h2>"
    
    # Coherence scores
    if coherence_summary:
        html += "<h3>Topic Coherence</h3>"
        html += "<p>Topic coherence measures how semantically similar the top words in each topic are. Higher is better.</p>"
        html += "<table>"
        html += "<tr><th>Model</th><th>Mean Coherence</th><th>Improvement</th></tr>"
        html += f"<tr><td class='lda'>LDA</td><td>{coherence_summary.get('lda_coherence_mean', 'N/A'):.4f}</td><td>-</td></tr>"
        html += f"<tr><td class='bert'>BERTopic</td><td>{coherence_summary.get('bert_coherence_mean', 'N/A'):.4f}</td>"
        
        improvement = coherence_summary.get('percent_improvement', 0)
        if improvement > 0:
            html += f"<td>+{improvement:.2f}%</td></tr>"
        else:
            html += f"<td>{improvement:.2f}%</td></tr>"
        
        html += "</table>"
        html += "<img src='coherence_comparison.png' alt='Coherence Comparison Chart'>"
    
    # User narrowness metrics
    if narrowness_summary:
        html += "<h3>User Topic Narrowness Correlation</h3>"
        html += "<p>This shows how consistently each model identifies user topic concentration.</p>"
        html += "<table>"
        html += "<tr><th>Metric</th><th>Correlation</th><th>LDA Mean</th><th>BERTopic Mean</th><th>Difference</th></tr>"
        
        for metric, data in narrowness_summary.items():
            html += f"<tr><td>{metric.replace('_', ' ').title()}</td>"
            html += f"<td>{data.get('correlation', 'N/A'):.3f}</td>"
            html += f"<td>{data.get('lda_mean', 'N/A'):.3f}</td>"
            html += f"<td>{data.get('bert_mean', 'N/A'):.3f}</td>"
            
            diff = data.get('mean_difference', 0)
            if diff > 0:
                html += f"<td>+{diff:.3f}</td></tr>"
            else:
                html += f"<td>{diff:.3f}</td></tr>"
        
        html += "</table>"
        html += "<img src='narrowness_metrics_comparison.png' alt='Narrowness Metrics Comparison'>"
    
    # Suspicious user detection
    if suspicious_summary:
        html += "<h3>Suspicious User Detection</h3>"
        html += "<p>This compares how each model identifies suspicious or anomalous users.</p>"
        html += "<table>"
        html += "<tr><th>Metric</th><th>Value</th></tr>"
        html += f"<tr><td>Total Suspicious Users (LDA)</td><td>{suspicious_summary.get('total_suspicious_lda', 'N/A')}</td></tr>"
        html += f"<tr><td>Total Suspicious Users (BERTopic)</td><td>{suspicious_summary.get('total_suspicious_bert', 'N/A')}</td></tr>"
        html += f"<tr><td>Overlap Count</td><td>{suspicious_summary.get('overlap_count', 'N/A')}</td></tr>"
        html += f"<tr><td>Overlap % of LDA</td><td>{suspicious_summary.get('overlap_percent_of_lda', 'N/A'):.1f}%</td></tr>"
        html += f"<tr><td>Overlap % of BERTopic</td><td>{suspicious_summary.get('overlap_percent_of_bert', 'N/A'):.1f}%</td></tr>"
        html += f"<tr><td>LDA-only Suspicious Users</td><td>{suspicious_summary.get('lda_only_count', 'N/A')}</td></tr>"
        html += f"<tr><td>BERTopic-only Suspicious Users</td><td>{suspicious_summary.get('bert_only_count', 'N/A')}</td></tr>"
        html += "</table>"
        html += "<img src='suspicious_users_overlap.png' alt='Suspicious Users Overlap'>"
    
    html += "</div>"
    
    # Topic Similarity Analysis
    if topic_similarities:
        html += "<div class='section'>"
        html += "<h2>Topic Similarity Analysis</h2>"
        html += "<p>This shows which topics from different models are most similar based on shared words.</p>"
        html += "<table>"
        html += "<tr><th>LDA Topic</th><th>BERTopic Topic</th><th>Similarity</th><th>Shared Words</th></tr>"
        
        for sim in topic_similarities[:5]:  # Show top 5 similar pairs
            html += f"<tr>"
            html += f"<td class='lda'>Topic {sim['lda_topic']}</td>"
            html += f"<td class='bert'>Topic {sim['bert_topic']}</td>"
            html += f"<td>{sim['similarity']:.3f}</td>"
            html += f"<td>{', '.join(sim['shared_words'][:5])}</td>"
            html += f"</tr>"
        
        html += "</table>"
        html += "<p>See the <a href='topic_words_comparison.html'>detailed topic words comparison</a> for more information.</p>"
        html += "</div>"
    
    # Conclusions
    html += "<div class='section'>"
    html += "<h2>Conclusions</h2>"
    
    # Generate some basic conclusions based on the metrics
    conclusions = []
    
    # Coherence conclusion
    if coherence_summary:
        if coherence_summary.get('difference', 0) > 0:
            conclusions.append("BERTopic produced more coherent topics than LDA, which suggests that the embedding-based approach may better capture semantic relationships.")
        else:
            conclusions.append("LDA produced more coherent topics than BERTopic, suggesting that for this dataset, the traditional probabilistic approach works well.")
    
    # Narrowness metrics conclusion
    if narrowness_summary and 'gini_coefficient' in narrowness_summary:
        corr = narrowness_summary['gini_coefficient'].get('correlation', 0)
        if corr > 0.7:
            conclusions.append("There is strong agreement between models on user topic narrowness, suggesting that both approaches are capturing similar user behavior patterns.")
        elif corr > 0.4:
            conclusions.append("There is moderate agreement between models on user topic narrowness, indicating some consistency but also different perspectives on user behavior.")
        else:
            conclusions.append("There is weak agreement between models on user topic narrowness, suggesting that the approaches are capturing different aspects of user behavior.")
    
    # Suspicious detection conclusion
    if suspicious_summary:
        overlap_percent = suspicious_summary.get('overlap_percent_of_lda', 0)
        if overlap_percent > 70:
            conclusions.append("There is high agreement on suspicious user detection, suggesting that anomalous behavior is consistently identifiable regardless of the topic modeling approach.")
        elif overlap_percent > 40:
            conclusions.append("There is moderate agreement on suspicious user detection, suggesting that combining both approaches might provide more robust detection.")
        else:
            conclusions.append("There is low agreement on suspicious user detection, suggesting that each model is sensitive to different types of anomalous patterns.")
    
    # Overall conclusion
    if coherence_summary and coherence_summary.get('difference', 0) > 0:
        conclusions.append("Overall, BERTopic appears to provide better topic quality for this dataset, but both models offer valuable and complementary perspectives on the data.")
    else:
        conclusions.append("Overall, both models offer valuable and complementary perspectives on the data, with different strengths depending on the specific analysis goal.")
    
    # Add conclusions to HTML
    html += "<ul>"
    for conclusion in conclusions:
        html += f"<li>{conclusion}</li>"
    html += "</ul>"
    
    html += "</div>"
    
    # Footer
    html += "<div class='section'>"
    html += "<p><em>Report generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "</em></p>"
    html += "</div>"
    
    html += "</body></html>"
    
    # Save HTML report
    with open(os.path.join(output_dir, "model_comparison_report.html"), 'w') as f:
        f.write(html)
    
    print(f"Comparison report generated at {os.path.join(output_dir, 'model_comparison_report.html')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare LDA and BERTopic models on the same dataset')
    parser.add_argument('input_file', help='Path to input data file')
    parser.add_argument('output_dir', help='Directory to save comparison outputs')
    parser.add_argument('--lda-topics', type=int, default=20, help='Number of topics for LDA model')
    parser.add_argument('--bert-topics', type=int, default=20, help='Target number of topics for BERTopic model')
    parser.add_argument('--combine', action='store_true', help='Combine posts into time windows')
    
    args = parser.parse_args()
    
    run_comparison(
        input_file=args.input_file,
        output_dir=args.output_dir,
        lda_topics=args.lda_topics,
        bert_topics=args.bert_topics,
        combine_by_window=args.combine
    )
