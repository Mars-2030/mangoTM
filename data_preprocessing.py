# data_preprocessing.py

import pandas as pd
import datetime
from pathlib import Path
import argparse

# --- Keep existing preprocess_reddit_data and preprocess_fake_data functions ---

def preprocess_reddit_data(input_file, output_file):
    """
    Preprocess Reddit data to match the topic modeling system's expected format.

    Args:
        input_file: Path to the reddit_vm.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading Reddit data from {input_file}...")
    try:
        reddit_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None

    print(f"Loaded {len(reddit_df)} Reddit posts")
    print(f"Columns: {reddit_df.columns.tolist()}")

    processed_df = pd.DataFrame()
    if 'id' not in reddit_df.columns:
        print("Error: 'id' column missing in Reddit data.")
        return None
    processed_df['user_id'] = reddit_df['id'].astype(str) # Ensure user_id is string

    if 'timestamp' in reddit_df.columns:
        processed_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'], errors='coerce')
    elif 'created' in reddit_df.columns:
        processed_df['timestamp'] = pd.to_datetime(reddit_df['created'], unit='s', errors='coerce')
    else:
         print("Error: No recognizable timestamp column ('timestamp' or 'created') found in Reddit data.")
         return None

    # Combine title and body for post_content
    title_col = reddit_df.get('title', pd.Series(dtype=str)).fillna('')
    body_col = reddit_df.get('body', pd.Series(dtype=str)).fillna('')
    processed_df['post_content'] = (title_col + " " + body_col).str.strip()


    processed_df = processed_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    processed_df = processed_df[processed_df['post_content'] != ""]

    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed Reddit data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")

    return processed_df

def preprocess_fake_data(input_file, output_file):
    """
    Preprocess fake data to match the topic modeling system's expected format.

    Args:
        input_file: Path to the fake_data.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading fake data from {input_file}...")
    try:
        fake_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None

    print(f"Loaded {len(fake_df)} fake posts")
    print(f"Columns: {fake_df.columns.tolist()}")

    processed_df = pd.DataFrame()

    if 'username' in fake_df.columns:
        processed_df['user_id'] = fake_df['username'].astype(str)
    elif 'id' in fake_df.columns:
         processed_df['user_id'] = fake_df['id'].astype(str)
    else:
        print("Error: No recognizable user identifier ('username' or 'id') found in fake data.")
        return None

    if 'timestamp' in fake_df.columns:
        processed_df['timestamp'] = pd.to_datetime(fake_df['timestamp'], errors='coerce')
    else:
        # Create timestamp if it doesn't exist
        base_timestamp = datetime.datetime.now(datetime.timezone.utc) # Use timezone-aware
        timestamps = [base_timestamp - datetime.timedelta(days=int(60 * i/len(fake_df)))
                     for i in range(len(fake_df))]
        processed_df['timestamp'] = timestamps
        print("Warning: 'timestamp' column not found. Generated synthetic timestamps.")


    if 'text' in fake_df.columns:
        processed_df['post_content'] = fake_df['text']
    elif 'body' in fake_df.columns: # Added flexibility
         processed_df['post_content'] = fake_df['body']
    else:
        print("Error: No recognizable content column ('text' or 'body') found in fake data.")
        return None

    processed_df = processed_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    processed_df = processed_df[processed_df['post_content'] != ""]

    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed fake data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")

    return processed_df

# --- NEW FUNCTION ---
def preprocess_ukraine_russia_data(input_file, output_file):
    """
    Preprocess ukraine_russia Twitter data to match the topic modeling system's expected format.

    Args:
        input_file: Path to the ukraine_russia.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading Ukraine/Russia Twitter data from {input_file}...")
    try:
        # Use low_memory=False for potentially mixed types, adjust if needed
        twitter_df = pd.read_csv(input_file, low_memory=False)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None


    print(f"Loaded {len(twitter_df)} tweets")
    print(f"Columns: {twitter_df.columns.tolist()}")

    required_columns = ['user_id', 'created_at', 'tweet']
    missing_cols = [col for col in required_columns if col not in twitter_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in {input_file}: {missing_cols}")
        return None

    processed_df = pd.DataFrame()

    # Map columns
    processed_df['user_id'] = twitter_df['user_id'].astype(str) # Ensure user_id is string
    processed_df['timestamp'] = pd.to_datetime(twitter_df['created_at'], errors='coerce')
    processed_df['post_content'] = twitter_df['tweet']

    # Drop rows with missing essential data
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    processed_df = processed_df[processed_df['post_content'].astype(str).str.strip() != ""]
    dropped_rows = initial_rows - len(processed_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing user_id, timestamp, or empty content.")


    # Save to CSV
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed Ukraine/Russia Twitter data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")

    return processed_df
# --- END NEW FUNCTION ---

def preprocess_troll_tweet_data(input_file, output_file):
    """
    Preprocess streamlined_tweets1.csv (Russian Troll data) to match
    the topic modeling system's expected format.

    Args:
        input_file: Path to the streamlined_tweets1.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading Troll Tweet data from {input_file}...")
    try:
        # Skip the first 3 rows of metadata/comments
        # Specify column names explicitly based on the 4th line
        col_names = ['screenname', 'date_sent', 'text', 'retweets', 'favorites']
        # Use 'header=None' since we skip rows and provide names
        # Use 'on_bad_lines' to skip problematic lines if any
        troll_df = pd.read_csv(input_file, skiprows=3, header=None, names=col_names,
                              usecols=[0, 1, 2], # Only read the first 3 columns needed
                              on_bad_lines='warn', # Or 'skip'
                              low_memory=False) # Add low_memory=False
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"Error loading Troll Tweet CSV: {e}")
        # e.g., if columns don't match expected after skiprows
        return None

    print(f"Loaded {len(troll_df)} troll tweets (initial read)")
    print(f"Columns used: {troll_df.columns.tolist()}")

    processed_df = pd.DataFrame()

    # Check if expected columns were loaded correctly
    if 'screenname' not in troll_df.columns or 'date_sent' not in troll_df.columns or 'text' not in troll_df.columns:
         print(f"Error: Could not find expected columns ('screenname', 'date_sent', 'text') after skipping rows in {input_file}.")
         print(f"Actual columns found: {troll_df.columns.tolist()}")
         return None


    # Map columns to standard names
    processed_df['user_id'] = troll_df['screenname'].astype(str)
    # Parse the timestamp, handling potential variations and making it UTC
    processed_df['timestamp'] = pd.to_datetime(troll_df['date_sent'], errors='coerce', utc=True)
    processed_df['post_content'] = troll_df['text']

    # Ensure content is string and fill NaNs
    processed_df['post_content'] = processed_df['post_content'].astype(str).fillna('')

    # Drop rows with missing essential data after processing
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    processed_df = processed_df[processed_df['post_content'].str.strip() != ""] # Filter empty strings
    dropped_rows = initial_rows - len(processed_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing user_id, timestamp, or empty/invalid content.")

    # Timestamp should already be UTC from parsing with utc=True

    # Save to CSV
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed Troll Tweet data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")

    return processed_df
# --- END NEW FUNCTION ---
# --- NEW FUNCTION for streamlined_tweets1.csv ---
def preprocess_streamlined_tweets(input_file, output_file):
    """
    Preprocess streamlined_tweets1.csv data to match the topic modeling system's expected format.

    Args:
        input_file: Path to the streamlined_tweets1.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading Streamlined Tweets data from {input_file}...")
    try:
        # Read CSV, assume extra columns are unnamed and can be dropped later
        tweets_df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        return None
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    print(f"Loaded {len(tweets_df)} tweets")
    print(f"Columns: {tweets_df.columns.tolist()}")

    # Identify the columns we need based on their names in the sample
    user_col = 'Twitter screenname'
    timestamp_col = 'Date tweet sent'
    content_col = 'Tweet text'

    required_columns = [user_col, timestamp_col, content_col]
    missing_cols = [col for col in required_columns if col not in tweets_df.columns]
    if missing_cols:
        print(f"Error: Missing required columns in {input_file}: {missing_cols}")
        return None

    processed_df = pd.DataFrame()

    # Map columns
    processed_df['user_id'] = tweets_df[user_col].astype(str) # Ensure user_id is string
    processed_df['timestamp'] = pd.to_datetime(tweets_df[timestamp_col], errors='coerce')
    processed_df['post_content'] = tweets_df[content_col]

    # Drop rows with missing essential data after mapping and conversion
    initial_rows = len(processed_df)
    processed_df = processed_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    # Also drop rows where content might be NaN *before* stripping
    processed_df = processed_df[processed_df['post_content'].notna()]
    processed_df = processed_df[processed_df['post_content'].astype(str).str.strip() != ""]
    dropped_rows = initial_rows - len(processed_df)
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows due to missing user_id, timestamp, or empty/NaN content.")


    # Save to CSV
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed Streamlined Tweets data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")

    return processed_df
# --- END NEW FUNCTION ---


def combine_datasets(file_list, output_file):
    """
    Combine multiple preprocessed datasets into a single dataset.
    (Existing function - unchanged)
    """
    print(f"Combining datasets: {', '.join(file_list)}...")
    all_dfs = []
    for file_path in file_list:
        try:
            df = pd.read_csv(file_path)
            # Ensure timestamp is datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['user_id'] = df['user_id'].astype(str) # Ensure user_id is string
            df = df.dropna(subset=['user_id', 'timestamp', 'post_content']) # Re-check NAs
            df = df[df['post_content'].astype(str).str.strip() != ""] # Re-check empty content
            all_dfs.append(df[['user_id', 'timestamp', 'post_content']]) # Keep only essential columns
        except FileNotFoundError:
            print(f"Warning: File not found during combination: {file_path}")
        except Exception as e:
             print(f"Warning: Could not load or process {file_path} during combination: {e}")

    if not all_dfs:
        print("Error: No valid dataframes to combine.")
        return None

    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')

    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")
    print(f"Final combined dataset has {len(combined_df)} posts")

    return combined_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for topic modeling')
    # Added 'streamlined_tweets' and allow multiple selections or 'all'
    parser.add_argument('--datasets', nargs='+', default=['reddit', 'fake'],
                        help='Datasets to process (e.g., reddit fake ukraine streamlined_tweets) or "all"')
    parser.add_argument('--combine', action='store_true',
                        help='Combine the processed datasets into one file')
    parser.add_argument('--input-dir', default='data', help='Directory containing raw data files (changed default to current)')
    parser.add_argument('--output-dir', default='preprocessed_data', help='Directory to save preprocessed files')

    args = parser.parse_args()

    # Define dataset configurations - Added 'streamlined_tweets'
    dataset_configs = {
        'reddit': {'input': 'reddit_vm.csv', 'func': preprocess_reddit_data, 'output': 'reddit_processed.csv'},
        'fake': {'input': 'fake_data.csv', 'func': preprocess_fake_data, 'output': 'fake_processed.csv'},
        'ukraine': {'input': 'ukraine_russia.csv', 'func': preprocess_ukraine_russia_data, 'output': 'ukraine_russia_processed.csv'},
        'streamlined_tweets': {'input': 'streamlined_tweets1.csv', 'func': preprocess_streamlined_tweets, 'output': 'streamlined_tweets_processed.csv'}
    }

    # Determine which datasets to process
    if 'all' in args.datasets:
        datasets_to_process = list(dataset_configs.keys())
    else:
        datasets_to_process = [ds for ds in args.datasets if ds in dataset_configs]
        if not datasets_to_process:
            print("Error: No valid datasets specified. Choose from:", list(dataset_configs.keys()), 'or "all"')
            return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    input_dir = Path(args.input_dir)

    processed_files = []

    print(f"Processing datasets: {', '.join(datasets_to_process)}")
    for ds_name in datasets_to_process:
        config = dataset_configs[ds_name]
        input_path = input_dir / config['input']
        output_path = output_dir / config['output']

        print(f"\n--- Processing {ds_name} ---")
        processed_df = config['func'](input_path, output_path)
        if processed_df is not None and not processed_df.empty: # Check if df is not None and not empty
            processed_files.append(str(output_path)) # Store path for potential combination
        else:
            print(f"Failed to process {ds_name} or resulted in empty dataset.")


    print("\n--- Individual Dataset Processing Complete ---")
    for ds_name in datasets_to_process:
         config = dataset_configs[ds_name]
         output_path = output_dir / config['output']
         if str(output_path) in processed_files:
             # Check if the file actually exists and has data
             if output_path.exists() and output_path.stat().st_size > 50: # Check size > header approx
                 print(f"* {ds_name}: Use {output_path}")
             else:
                 print(f"* {ds_name}: Processed, but output file is empty or missing ({output_path}).")
         else:
             print(f"* {ds_name}: Processing failed or skipped.")


    # Combine datasets if requested and multiple files were processed successfully
    if args.combine and len(processed_files) >= 1: # Allow combining even if only one was processed
        print("\n--- Combining Processed Datasets ---")
        combined_output_path = output_dir / "combined_data.csv"
        combine_datasets(processed_files, combined_output_path)
        # Check if combined file was created and has data
        if combined_output_path.exists() and combined_output_path.stat().st_size > 50:
             print(f"\nCombined data available at: {combined_output_path}")
        else:
             print(f"\nWarning: Combination process completed, but the output file is empty or missing: {combined_output_path}")

    elif args.combine and len(processed_files) == 0:
        print("\nSkipping combination: No datasets were successfully processed.")


    print("\nData preprocessing finished!")

if __name__ == "__main__":
    main()