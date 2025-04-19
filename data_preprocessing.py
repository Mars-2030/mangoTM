import pandas as pd
import datetime
from pathlib import Path
import argparse


def preprocess_reddit_data(input_file, output_file):
    """
    Preprocess Reddit data to match the topic modeling system's expected format.
    
    Args:
        input_file: Path to the reddit_vm.csv file
        output_file: Path to save the preprocessed data
    """
    print(f"Loading Reddit data from {input_file}...")
    reddit_df = pd.read_csv(input_file)
    
    # Print basic stats
    print(f"Loaded {len(reddit_df)} Reddit posts")
    print(f"Columns: {reddit_df.columns.tolist()}")
    
    # Create a new DataFrame with the required columns
    processed_df = pd.DataFrame()
    
    # Use id as user_id (since we don't have actual user information)
    processed_df['user_id'] = reddit_df['id']
    
    # Convert timestamp or created to proper datetime
    if 'timestamp' in reddit_df.columns:
        processed_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
    else:
        # Convert 'created' which is likely a Unix timestamp
        processed_df['timestamp'] = pd.to_datetime(reddit_df['created'], unit='s')
    
    # Combine title and body for post_content
    if 'body' in reddit_df.columns and 'title' in reddit_df.columns:
        # If both title and body exist, combine them
        processed_df['post_content'] = reddit_df['title'] + " " + reddit_df['body'].fillna("")
    elif 'title' in reddit_df.columns:
        # If only title exists
        processed_df['post_content'] = reddit_df['title']
    elif 'body' in reddit_df.columns:
        # If only body exists
        processed_df['post_content'] = reddit_df['body']
    
    # Drop rows with empty post_content
    processed_df = processed_df[processed_df['post_content'].notna() & (processed_df['post_content'] != "")]
    
    # Save to CSV
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
    fake_df = pd.read_csv(input_file)
    
    # Print basic stats
    print(f"Loaded {len(fake_df)} fake posts")
    print(f"Columns: {fake_df.columns.tolist()}")
    
    # Create a new DataFrame with the required columns
    processed_df = pd.DataFrame()
    
    # Use username as user_id
    if 'username' in fake_df.columns:
        processed_df['user_id'] = fake_df['username']
    else:
        # If username doesn't exist, use id
        processed_df['user_id'] = fake_df['id'].astype(str)
    
    # Create timestamp if it doesn't exist (use current time with some variation)
    if 'timestamp' not in fake_df.columns:
        base_timestamp = datetime.datetime.now()
        # Generate timestamps ranging from 60 days ago to today
        timestamps = [base_timestamp - datetime.timedelta(days=int(60 * i/len(fake_df))) 
                     for i in range(len(fake_df))]
        processed_df['timestamp'] = timestamps
    else:
        processed_df['timestamp'] = pd.to_datetime(fake_df['timestamp'])
    
    # Use text as post_content
    processed_df['post_content'] = fake_df['text']
    
    # Drop rows with empty post_content
    processed_df = processed_df[processed_df['post_content'].notna() & (processed_df['post_content'] != "")]
    
    # Save to CSV
    processed_df.to_csv(output_file, index=False)
    print(f"Preprocessed fake data saved to {output_file}")
    print(f"Final dataset has {len(processed_df)} posts")
    
    return processed_df

def combine_datasets(reddit_file, fake_file, output_file):
    """
    Combine preprocessed Reddit and fake data into a single dataset.
    
    Args:
        reddit_file: Path to preprocessed Reddit data
        fake_file: Path to preprocessed fake data
        output_file: Path to save the combined data
    """
    print(f"Combining datasets...")
    reddit_df = pd.read_csv(reddit_file)
    fake_df = pd.read_csv(fake_file)
    
    # Ensure timestamp is datetime
    reddit_df['timestamp'] = pd.to_datetime(reddit_df['timestamp'])
    fake_df['timestamp'] = pd.to_datetime(fake_df['timestamp'])
    
    # Combine the datasets
    combined_df = pd.concat([reddit_df, fake_df], ignore_index=True)
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined dataset saved to {output_file}")
    print(f"Final combined dataset has {len(combined_df)} posts")
    
    return combined_df

def main():
    parser = argparse.ArgumentParser(description='Preprocess data for topic modeling')
    parser.add_argument('--dataset', choices=['reddit', 'fake', 'combined'], 
                        default='combined', help='Dataset to process')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(exist_ok=True)
    
    if args.dataset == 'reddit' or args.dataset == 'combined':
        # Preprocess Reddit data
        reddit_processed = preprocess_reddit_data(
            "data/reddit_vm.csv", 
            output_dir / "reddit_processed.csv"
        )
    
    if args.dataset == 'fake' or args.dataset == 'combined':
        # Preprocess fake data
        fake_processed = preprocess_fake_data(
            "data/fake_data.csv", 
            output_dir / "fake_processed.csv"
        )
    
    if args.dataset == 'combined':
        # Combine datasets
        combined_data = combine_datasets(
            output_dir / "reddit_processed.csv",
            output_dir / "fake_processed.csv",
            output_dir / "combined_data.csv"
        )
    
    print("\nData preprocessing complete!")
    if args.dataset == 'reddit':
        print(f"Use this file path in your topic modeling code: {output_dir / 'reddit_processed.csv'}")
    elif args.dataset == 'fake':
        print(f"Use this file path in your topic modeling code: {output_dir / 'fake_processed.csv'}")
    else:
        print(f"Use this file path in your topic modeling code: {output_dir / 'combined_data.csv'}")

if __name__ == "__main__":
    main()
