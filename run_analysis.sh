#!/bin/bash
# Complete script to run the entire topic modeling pipeline
# Now uses a variable for the dataset name/analysis scope and time bin

# --- Configuration ---
# Define the dataset name or analysis scope here
# Examples: "fake", "reddit", "ukraine", "streamlined_tweets", "combined"
ANALYSIS_SCOPE="streamlined_tweets" # <<< CHANGE THIS VALUE TO RUN FOR A DIFFERENT DATASET

# Define the time binning strategy ('day', 'week', 'month')
TIME_BIN="day"          # <<< CHANGE THIS VALUE TO ADJUST TIME BINNING

# Define base directories
RAW_DATA_DIR="data"        # <<< Directory containing raw input CSVs
PREPROCESSED_DIR="preprocessed_data"
RESULTS_BASE_DIR="topic_analysis_results"

# Define the specific output directory for this analysis run
ANALYSIS_OUTPUT_DIR="${RESULTS_BASE_DIR}/${ANALYSIS_SCOPE}_analysis"

# --- Dataset Specific Settings ---
# Adjust these parameters based on the dataset characteristics
# You could also override TIME_BIN here per dataset if needed
if [ "$ANALYSIS_SCOPE" == "ukraine" ]; then
    PREPROCESSED_FILE="${PREPROCESSED_DIR}/ukraine_russia_processed.csv"
    DATASET_TO_PREPROCESS="ukraine"
    LDA_TOPICS=5
    BERT_MIN_SIZE=5
    BERT_TOPICS=5
    SKIP_BERTOPIC_FLAG=""
elif [ "$ANALYSIS_SCOPE" == "fake" ]; then
    PREPROCESSED_FILE="${PREPROCESSED_DIR}/fake_processed.csv"
    DATASET_TO_PREPROCESS="fake"
    LDA_TOPICS=5
    BERT_MIN_SIZE=5
    BERT_TOPICS=5
    SKIP_BERTOPIC_FLAG=""
elif [ "$ANALYSIS_SCOPE" == "reddit" ]; then
    PREPROCESSED_FILE="${PREPROCESSED_DIR}/reddit_processed.csv"
    DATASET_TO_PREPROCESS="reddit"
    LDA_TOPICS=5
    BERT_MIN_SIZE=5
    BERT_TOPICS=5
    SKIP_BERTOPIC_FLAG=""
elif [ "$ANALYSIS_SCOPE" == "streamlined_tweets" ]; then
    PREPROCESSED_FILE="${PREPROCESSED_DIR}/streamlined_tweets_processed.csv"
    DATASET_TO_PREPROCESS="streamlined_tweets"
    LDA_TOPICS=12
    BERT_MIN_SIZE=12
    BERT_TOPICS=12
    SKIP_BERTOPIC_FLAG=""
elif [ "$ANALYSIS_SCOPE" == "combined" ]; then
    PREPROCESSED_FILE="${PREPROCESSED_DIR}/combined_data.csv"
    DATASET_TO_PREPROCESS="" # Don't run individual preprocessing
    LDA_TOPICS=5
    BERT_MIN_SIZE=5
    BERT_TOPICS=5
    SKIP_BERTOPIC_FLAG=""
else
    echo "Error: Unknown ANALYSIS_SCOPE '$ANALYSIS_SCOPE'. Please set it correctly."
    exit 1
fi

# --- Setup ---
echo "Starting topic modeling pipeline for: $ANALYSIS_SCOPE"
echo "Output directory: $ANALYSIS_OUTPUT_DIR"
echo "Raw Data Directory: $RAW_DATA_DIR" # Added echo for raw data dir
echo "Using Preprocessed Input: $PREPROCESSED_FILE"
echo "Time Binning: $TIME_BIN"
echo "LDA Topics: $LDA_TOPICS"
echo "BERTopic Min Size: $BERT_MIN_SIZE"
echo "BERTopic Target Topics: $BERT_TOPICS"
echo "Skip BERTopic Flag: $SKIP_BERTOPIC_FLAG"


# Step 1: Create necessary directories
echo "Creating directories..."
mkdir -p "$PREPROCESSED_DIR"
mkdir -p "$ANALYSIS_OUTPUT_DIR"
mkdir -p "$ANALYSIS_OUTPUT_DIR/lda_results"
mkdir -p "$ANALYSIS_OUTPUT_DIR/bertopic_results"
mkdir -p "$ANALYSIS_OUTPUT_DIR/model_comparison"


# Step 2: Preprocess the data (unless it's 'combined')
if [ -n "$DATASET_TO_PREPROCESS" ]; then
    echo "Step 1/3: Preprocessing $DATASET_TO_PREPROCESS data..."
    # *** FIXED: Use the RAW_DATA_DIR variable for --input-dir ***
    python data_preprocessing.py --datasets "$DATASET_TO_PREPROCESS" --output-dir "$PREPROCESSED_DIR" --input-dir "$RAW_DATA_DIR" 
    # Check if the preprocessing succeeded
    if [ ! -f "$PREPROCESSED_FILE" ] || [ ! -s "$PREPROCESSED_FILE" ]; then # Check exists and is not empty
        echo "Error: Preprocessing failed. Expected file '$PREPROCESSED_FILE' not found or is empty."
        exit 1
    fi
elif [ "$ANALYSIS_SCOPE" == "combined" ]; then
     echo "Step 1/3: Skipping preprocessing for 'combined' scope."
     echo "Ensure '$PREPROCESSED_FILE' exists from a previous run with e.g. '--datasets all --combine'."
     if [ ! -f "$PREPROCESSED_FILE" ] || [ ! -s "$PREPROCESSED_FILE" ]; then # Check exists and is not empty
        echo "Error: Combined file '$PREPROCESSED_FILE' not found or is empty. Please run preprocessing first."
        # Suggestion for how to create the combined file - Updated to use RAW_DATA_DIR
        echo "You might need to run: python data_preprocessing.py --datasets reddit fake ukraine streamlined_tweets --combine --output-dir $PREPROCESSED_DIR --input-dir $RAW_DATA_DIR"
        exit 1
     fi
fi

# Step 3: Run topic analysis using dynamically built arguments
echo "Step 2/3: Running topic analysis..."

# Build the command arguments conditionally in an array (safer)
CMD_ARGS=(
    --input "$PREPROCESSED_FILE"
    --output "$ANALYSIS_OUTPUT_DIR"
    --lda-topics "$LDA_TOPICS"
    --bert-min-size "$BERT_MIN_SIZE"
    --time-bin "$TIME_BIN"
)

# Conditionally add --bert-topics if BERT_TOPICS has a non-empty value
if [ -n "$BERT_TOPICS" ]; then
    CMD_ARGS+=(--bert-topics "$BERT_TOPICS")
fi

# Conditionally add the skip flag if SKIP_BERTOPIC_FLAG has a non-empty value
if [ -n "$SKIP_BERTOPIC_FLAG" ]; then
    CMD_ARGS+=("$SKIP_BERTOPIC_FLAG")
fi

# Execute the command with the constructed arguments
# Using "${CMD_ARGS[@]}" ensures proper handling of arguments
echo "Executing: python topic_analysis.py ${CMD_ARGS[@]}" # Print command for debugging
python topic_analysis.py "${CMD_ARGS[@]}"
ANALYSIS_EXIT_CODE=$? # Capture exit code

if [ $ANALYSIS_EXIT_CODE -ne 0 ]; then
    echo "Error: Topic analysis script failed with exit code $ANALYSIS_EXIT_CODE."
    exit $ANALYSIS_EXIT_CODE
fi


# Step 4: Generate notebook for interactive exploration (Optional)
echo "Step 3/3: Skipping interactive visualization preparation."
# Example: You might copy a template notebook and potentially inject variables later
# jupyter notebook visualization_notebook.ipynb || echo "Jupyter notebook command failed or not installed."

echo "" # Newline for clarity
echo "Analysis pipeline complete for '$ANALYSIS_SCOPE'!"
echo "Results saved to:"
echo "  - Raw Data location assumed: $RAW_DATA_DIR/"
echo "  - Preprocessed data: $PREPROCESSED_DIR/"
echo "  - Analysis results: $ANALYSIS_OUTPUT_DIR/"
# Example of how to print specific sub-folders:
echo "    - LDA results: $ANALYSIS_OUTPUT_DIR/lda_results/"
echo "    - BERTopic results: $ANALYSIS_OUTPUT_DIR/bertopic_results/ (if run)"
echo "    - Model comparison: $ANALYSIS_OUTPUT_DIR/model_comparison/ (if run)"

# echo "To explore the results interactively, run: jupyter notebook visualization_notebook.ipynb"