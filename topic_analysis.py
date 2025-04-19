import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os
import argparse

# Add the directory containing the topic modeling code to the Python path
# Adjust this path if your topic modeling code is in a different location
sys.path.append('.')

# Import the topic modeling systems
from user_topic_modeling import UserTopicModelingSystem
from bertopic_extension import BERTopicModelingSystem


def analyze_with_lda(input_file, output_dir, num_topics=15, time_bin='week', combine_by_window=True):
    """
    Analyze the dataset using LDA topic modeling.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the results
        num_topics: Number of topics for LDA
        time_bin: Time binning for temporal analysis ('day', 'week', 'month')
        combine_by_window: Whether to combine posts into time windows
    """
    print(f"\n{'='*50}")
    print(f"Running LDA Topic Analysis with {num_topics} topics")
    print(f"{'='*50}")
    
    # Initialize the LDA system
    lda_system = UserTopicModelingSystem(
        num_topics=num_topics,
        time_bin=time_bin,
        lemmatize=True,
        extra_stopwords=['reddit', 'post', 'comment', 'www', 'http', 'https', 'com']
    )
    
    # Run the pipeline
    lda_results = lda_system.run_full_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        combine_by_window=combine_by_window,
        visualize_topics=True,
        detect_suspicious=True
    )
    
    print("\nLDA Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Analyzed {lda_results['num_documents']} documents from {lda_results['num_users']} users")
    print(f"Generated {lda_results['num_topics']} topics")
    
    return lda_system, lda_results


def analyze_with_bertopic(input_file, output_dir, min_topic_size=10, time_bin='week', combine_by_window=True):
    """
    Analyze the dataset using BERTopic.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save the results
        min_topic_size: Minimum topic size for BERTopic
        time_bin: Time binning for temporal analysis ('day', 'week', 'month')
        combine_by_window: Whether to combine posts into time windows
    """
    print(f"\n{'='*50}")
    print(f"Running BERTopic Analysis with min_topic_size={min_topic_size}")
    print(f"{'='*50}")
    
    # Initialize the BERTopic system
    bert_system = BERTopicModelingSystem(
        language_model='all-MiniLM-L6-v2',  # Lightweight model for faster processing
        time_bin=time_bin,
        min_topic_size=min_topic_size,
        extra_stopwords=['reddit', 'post', 'comment', 'www', 'http', 'https', 'com'],
        n_neighbors=15,
        n_components=5,
        min_cluster_size=min_topic_size
    )
    
    # Run the pipeline
    bert_results = bert_system.run_full_pipeline(
        input_file=input_file,
        output_dir=output_dir,
        combine_by_window=combine_by_window,
        visualize_topics=True,
        detect_suspicious=True
    )
    
    print("\nBERTopic Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Analyzed {bert_results['num_documents']} documents from {bert_results['num_users']} users")
    print(f"Generated {bert_results['num_topics']} topics")
    
    return bert_system, bert_results


def compare_models(lda_dir, bert_dir, output_dir):
    """
    Compare LDA and BERTopic results.
    
    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save comparison results
    """
    print(f"\n{'='*50}")
    print(f"Comparing LDA and BERTopic Results")
    print(f"{'='*50}")
    
    # Import the comparison script
    from comparison_script import compare_model_results
    
    # Run the comparison
    compare_model_results(lda_dir, bert_dir, output_dir)
    
    print("\nComparison Complete!")
    print(f"Comparison report saved to: {output_dir}/model_comparison_report.html")


def analyze_suspicious_users(lda_dir, bert_dir, output_dir):
    """
    Perform detailed analysis of suspicious users detected by both models.
    
    Args:
        lda_dir: Directory with LDA results
        bert_dir: Directory with BERTopic results
        output_dir: Directory to save suspicious user analysis
    """
    print(f"\n{'='*50}")
    print(f"Analyzing Suspicious Users")
    print(f"{'='*50}")
    
    # Load suspicious users data
    lda_suspicious = pd.read_csv(f"{lda_dir}/suspicious_users.csv")
    bert_suspicious = pd.read_csv(f"{bert_dir}/suspicious_users.csv")
    
    # Filter to only suspicious users
    lda_suspicious = lda_suspicious[lda_suspicious['suspicious']]
    bert_suspicious = bert_suspicious[bert_suspicious['suspicious']]
    
    print(f"LDA identified {len(lda_suspicious)} suspicious users")
    print(f"BERTopic identified {len(bert_suspicious)} suspicious users")
    
    # Find users identified by both models
    common_users = set(lda_suspicious['user_id']).intersection(set(bert_suspicious['user_id']))
    print(f"Users identified by both models: {len(common_users)}")
    
    # Create a report
    with open(f"{output_dir}/suspicious_users_report.md", 'w') as f:
        f.write("# Suspicious Users Analysis\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- LDA identified {len(lda_suspicious)} suspicious users\n")
        f.write(f"- BERTopic identified {len(bert_suspicious)} suspicious users\n")
        f.write(f"- {len(common_users)} users were identified by both models\n\n")
        
        f.write("## Common Suspicious Users\n\n")
        f.write("| User ID | Post Count | Gini (LDA) | Gini (BERTopic) | Top Topic (LDA) | Top Topic (BERTopic) |\n")
        f.write("|---------|------------|------------|-----------------|----------------|--------------------|\n")
        
        for user_id in common_users:
            lda_user = lda_suspicious[lda_suspicious['user_id'] == user_id].iloc[0]
            bert_user = bert_suspicious[bert_suspicious['user_id'] == user_id].iloc[0]
            
            f.write(f"| {user_id} | {lda_user['post_count']} | {lda_user['gini_coefficient']:.3f} | ")
            f.write(f"{bert_user['gini_coefficient']:.3f} | Topic {lda_user['dominant_topic']} | ")
            f.write(f"Topic {bert_user['dominant_topic']} |\n")
        
        f.write("\n## Suspicious Users - LDA Only\n\n")
        lda_only = set(lda_suspicious['user_id']) - set(bert_suspicious['user_id'])
        f.write(f"Number of users: {len(lda_only)}\n\n")
        
        if len(lda_only) > 0:
            f.write("| User ID | Post Count | Gini Coefficient | Top Topic | Duplicate Ratio |\n")
            f.write("|---------|------------|------------------|-----------|----------------|\n")
            
            for user_id in lda_only:
                user = lda_suspicious[lda_suspicious['user_id'] == user_id].iloc[0]
                f.write(f"| {user_id} | {user['post_count']} | {user['gini_coefficient']:.3f} | ")
                f.write(f"Topic {user['dominant_topic']} | {user['duplicate_post_ratio']:.3f} |\n")
        
        f.write("\n## Suspicious Users - BERTopic Only\n\n")
        bert_only = set(bert_suspicious['user_id']) - set(lda_suspicious['user_id'])
        f.write(f"Number of users: {len(bert_only)}\n\n")
        
        if len(bert_only) > 0:
            f.write("| User ID | Post Count | Gini Coefficient | Top Topic | Duplicate Ratio |\n")
            f.write("|---------|------------|------------------|-----------|----------------|\n")
            
            for user_id in bert_only:
                user = bert_suspicious[bert_suspicious['user_id'] == user_id].iloc[0]
                f.write(f"| {user_id} | {user['post_count']} | {user['gini_coefficient']:.3f} | ")
                f.write(f"Topic {user['dominant_topic']} | {user['duplicate_post_ratio']:.3f} |\n")
    
    print(f"Suspicious users report saved to: {output_dir}/suspicious_users_report.md")


def main():
    parser = argparse.ArgumentParser(description='Run topic modeling analysis on preprocessed data')
    parser.add_argument('--input', default='preprocessed_data/combined_data.csv',
                        help='Path to the preprocessed CSV file')
    parser.add_argument('--output', default='topic_analysis_results',
                        help='Directory to save analysis results')
    parser.add_argument('--lda-topics', type=int, default=15,
                        help='Number of topics for LDA model')
    parser.add_argument('--bert-min-size', type=int, default=10,
                        help='Minimum topic size for BERTopic model')
    parser.add_argument('--time-bin', choices=['day', 'week', 'month'], default='week',
                        help='Time binning for temporal analysis')
    parser.add_argument('--no-combine', action='store_true',
                        help='Do not combine posts into time windows')
    parser.add_argument('--skip-lda', action='store_true',
                        help='Skip LDA analysis')
    parser.add_argument('--skip-bertopic', action='store_true',
                        help='Skip BERTopic analysis')
    parser.add_argument('--skip-comparison', action='store_true',
                        help='Skip model comparison')
    
    args = parser.parse_args()
    
    # Create output directories
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    lda_dir = output_dir / "lda_results"
    lda_dir.mkdir(exist_ok=True)
    
    bert_dir = output_dir / "bertopic_results"
    bert_dir.mkdir(exist_ok=True)
    
    comparison_dir = output_dir / "model_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    # Run LDA analysis
    if not args.skip_lda:
        lda_system, lda_results = analyze_with_lda(
            input_file=args.input,
            output_dir=str(lda_dir),
            num_topics=args.lda_topics,
            time_bin=args.time_bin,
            combine_by_window=not args.no_combine
        )
    
    # Run BERTopic analysis
    if not args.skip_bertopic:
        bert_system, bert_results = analyze_with_bertopic(
            input_file=args.input,
            output_dir=str(bert_dir),
            min_topic_size=args.bert_min_size,
            time_bin=args.time_bin,
            combine_by_window=not args.no_combine
        )
    
    # Compare models
    if not args.skip_lda and not args.skip_bertopic and not args.skip_comparison:
        compare_models(str(lda_dir), str(bert_dir), str(comparison_dir))
        analyze_suspicious_users(str(lda_dir), str(bert_dir), str(comparison_dir))
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
