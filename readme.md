# Topic Modeling System for Reddit and Custom Data

This repository contains a topic modeling system designed to analyze user posting behavior from Reddit data (`reddit_vm.csv`) . The system uses LDA (Latent Dirichlet Allocation) to discover topic clusters, analyze user topic diversity, and detect suspicious behavior patterns.

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Data Files](#data-files)
- [File Structure](#file-structure)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [LDA Analysis](#lda-analysis)
  - [Analyzing Single Datasets](#analyzing-single-datasets)
  - [Full Pipeline](#full-pipeline)
- [Analysis Outputs](#analysis-outputs)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Advanced Topics](#advanced-topics)

## Overview

This system provides a complete pipeline for analyzing textual data, particularly focused on:

1. **Topic Discovery**: Identifying common themes across posts
2. **User Behavior Analysis**: Measuring how diverse or narrow each user's topics are
3. **Temporal Analysis**: Tracking topic evolution over time
4. **Suspicious User Detection**: Flagging outliers based on posting patterns

The primary algorithm used is Latent Dirichlet Allocation (LDA), a statistical topic modeling technique. The system also includes an optional BERTopic implementation for handling short texts through transformer-based embeddings.

## Installation

1. Clone this repository or download all the files
2. Install required packages:

```bash
pip install pandas numpy matplotlib seaborn nltk gensim scipy wordcloud scikit-learn
```

For optional components:
```bash
# For BERTopic (optional)
pip install bertopic sentence-transformers hdbscan

# For dimensionality reduction visualizations
pip install umap-learn

# For interactive visualizations
pip install jupyter ipywidgets
```

## Data Files

The system is set up to work with the following data files:

1. **reddit_vm.csv**: Contains 1566 Reddit posts with the following columns:
   - `title`: Post title
   - `score`: Post score/upvotes
   - `id`: Unique post identifier
   - `url`: Post URL
   - `comms_num`: Number of comments
   - `created`: Creation timestamp
   - `body`: Post content
   - `timestamp`: Formatted date/time



Make sure these files are placed in the project's root directory.

## File Structure

```
project_directory/
├── reddit_vm.csv                # Your Reddit data
├── fake_data.csv                # Your custom data
├── data_preprocessing.py        # Preprocessing script
├── topic_analysis.py            # Main analysis script
├── user_topic_modeling.py       # LDA implementation
├── bertopic_extension.py        # BERTopic implementation (optional)
├── comparison_script.py         # For comparing models (optional)
├── visualization_notebook.py    # Interactive visualization (optional)
├── run_analysis.sh              # Complete pipeline script
├── preprocessed_data/           # Will be created during execution
    ├── reddit_processed.csv     # Preprocessed Reddit data
    ├── fake_processed.csv       # Preprocessed custom data
    └── combined_data.csv        # Combined dataset
└── topic_analysis_results/      # Analysis outputs (will be created)
    ├── lda_results/             # LDA analysis results
    ├── bertopic_results/        # BERTopic results (if used)
    └── model_comparison/        # Comparison results (if both used)
```

## Usage

### Preprocessing

First, preprocess your data to prepare it for analysis:

```bash
# Process both datasets
python data_preprocessing.py

# Process single dataset
python data_preprocessing.py --dataset reddit  # For Reddit data only
python data_preprocessing.py --dataset fake    # For custom data only
```

This converts your raw data into the required format with `user_id`, `timestamp`, and `post_content` fields.

### LDA Analysis

Run the LDA topic modeling analysis:

```bash
python topic_analysis.py --input preprocessed_data/combined_data.csv --output topic_analysis_results --lda-topics 15 --skip-bertopic
```

Key parameters:
- `--input`: Path to the preprocessed data file
- `--output`: Directory to save analysis results
- `--lda-topics`: Number of topics for LDA (default: 15)
- `--skip-bertopic`: Skip BERTopic analysis (optional)

### Analyzing Single Datasets

To analyze each dataset separately:

```bash
# For Reddit data only
python data_preprocessing.py --dataset reddit
python topic_analysis.py --input preprocessed_data/reddit_processed.csv --output reddit_results --lda-topics 15 --skip-bertopic

# For custom data only
python data_preprocessing.py --dataset fake
python topic_analysis.py --input preprocessed_data/fake_processed.csv --output fake_results --lda-topics 10 --skip-bertopic
```

Note: For smaller datasets (like fake_data.csv), you may want to use fewer topics (e.g., 5-10) to avoid overfitting.

### Full Pipeline

For convenience, you can run the complete pipeline with a single command:

```bash
bash run_analysis.sh
```

This script:
1. Preprocesses both datasets
2. Runs LDA analysis
3. Generates visualization notebook (if Jupyter is installed)

## Analysis Outputs

The system generates several outputs in the specified output directory:

### 1. Data Files

- `user_topic_data.csv`: Complete user topic distributions and narrowness metrics
- `suspicious_users.csv`: Users flagged for suspicious behavior
- `topic_words.csv`: Top words for each discovered topic

### 2. Visualizations

- **Topic Word Clouds**: Visual representation of top words in each topic
- **User Topic Pie Charts**: Topic distribution for individual users
- **Temporal Evolution Charts**: Topic changes over time for users
- **Narrowness vs. Frequency Plot**: Scatter plot showing user topic concentration
- **Topic Embeddings**: 2D projection of topic relationships

### 3. Reports

- Summary statistics on topics and users
- Detailed analysis of suspicious users
- Comparison reports (if both LDA and BERTopic are used)

## Customization

You can customize various aspects of the analysis:

### Time Binning

Control how posts are grouped temporally:

```bash
python topic_analysis.py --time-bin day    # Group by day
python topic_analysis.py --time-bin week   # Group by week (default)
python topic_analysis.py --time-bin month  # Group by month
```

### Post Combination

Choose whether to combine multiple posts from the same user in a time period:

```bash
# Default: combine posts into time windows
python topic_analysis.py

# Don't combine posts (each post is analyzed separately)
python topic_analysis.py --no-combine
```

### Number of Topics

Adjust the number of topics based on your dataset size:

```bash
# For larger datasets (like reddit_vm.csv)
python topic_analysis.py --lda-topics 20

# For smaller datasets (like fake_data.csv)
python topic_analysis.py --lda-topics 10
```

### Additional Stopwords

You can customize the list of stopwords by modifying the `topic_analysis.py` file:

```python
# In topic_analysis.py, modify the extra_stopwords list
extra_stopwords=['reddit', 'post', 'comment', 'www', 'http', 'https', 'com', .....]
```

## Troubleshooting

### Common Issues

1. **Missing Columns Error**:
   ```
   ValueError: Missing required columns: ['user_id', 'timestamp', 'post_content']
   ```
   - Make sure your data has been properly preprocessed

2. **Empty Topics**:
   - Try reducing the number of topics: `--lda-topics 10`
   - Check that `min_post_length` isn't filtering too many posts

3. **Memory Errors**:
   - Reduce the dataset size or number of topics
   - Set `--skip-bertopic` to use only LDA, which is less memory-intensive

### Checking Preprocessing Results

Before running the full analysis, you can check if preprocessing worked correctly:

```python
import pandas as pd
# Check the preprocessed data
df = pd.read_csv('preprocessed_data/combined_data.csv')
print(df.head())
print(f"Number of posts: {len(df)}")
print(f"Number of users: {df['user_id'].nunique()}")
```

## Advanced Topics

### Adjusting LDA Parameters

You can modify the LDA parameters by editing the `user_topic_modeling.py` file:

```python
# In create_lda_model function
self.lda_model = LdaModel(
    corpus=self.corpus,
    id2word=self.dictionary,
    num_topics=self.num_topics,
    passes=30,  # Increase for better convergence
    alpha='auto',  # Or specify a value like 0.1
    eta='auto',   # Or specify a value like 0.01
    random_state=42
)
```

### Suspicious User Detection

Suspicious users are detected based on three criteria:
1. **High posting frequency**: Users in the top 5% of posting frequency
2. **Topic narrowness**: Users in the top 5% of Gini coefficient (topic concentration)
3. **Content duplication**: Users with high similarity between posts

You can adjust these thresholds in the `detect_suspicious_users` function.

### Custom Data Input Format

If you have data in a different format, you'll need to modify the `data_preprocessing.py` script to accommodate your specific data structure. The key requirements are:

1. Create a user identifier (`user_id`)
2. Convert timestamps to datetime format (`timestamp`)
3. Extract the text content (`post_content`)

### Interactive Visualization

For a more interactive exploration of the results, use the included Jupyter notebook:

```bash
jupyter notebook visualization_notebook.py
```

This notebook provides widgets to explore topics, user distributions, and suspicious patterns interactively.


