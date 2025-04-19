#!/bin/bash
# Complete script to run the entire topic modeling pipeline

# Step 1: Create necessary directories
mkdir -p preprocessed_data
mkdir -p topic_analysis_results
mkdir -p topic_analysis_results/lda_results
mkdir -p topic_analysis_results/bertopic_results
mkdir -p topic_analysis_results/model_comparison

echo "Starting topic modeling pipeline..."

# Step 2: Preprocess the data
echo "Step 1/3: Preprocessing data..."
python data_preprocessing.py

# Step 3: Run topic analysis
echo "Step 2/3: Running topic analysis..."
python topic_analysis.py --input preprocessed_data/combined_data.csv --output topic_analysis_results --lda-topics 15 --bert-min-size 10 --time-bin week

# Step 4: Generate notebook for interactive exploration
echo "Step 3/3: Preparing interactive visualization notebook..."
jupyter notebook visualization_notebook.ipynb || echo "Jupyter notebook opening skipped."

echo "Analysis pipeline complete!"
echo "Results saved to:"
echo "  - Preprocessed data: preprocessed_data/"
echo "  - LDA results: topic_analysis_results/lda_results/"
echo "  - BERTopic results: topic_analysis_results/bertopic_results/"
echo "  - Model comparison: topic_analysis_results/model_comparison/"
echo "  - Interactive notebook: visualization_notebook.ipynb (if Jupyter is installed)"
echo 
echo "To explore the results interactively, run: jupyter notebook visualization_notebook.ipynb"
