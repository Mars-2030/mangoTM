import pandas as pd
import json
from pathlib import Path
import re
from dateutil.parser import parse
from typing import List, Dict, Union, Optional
import argparse

def preprocess_jsonl_to_csv(input_file: str, output_file: str, 
                          user_id_field: str = 'user_id',
                          timestamp_field: str = 'created_at',
                          content_field: str = 'text',
                          additional_fields: Optional[List[str]] = None):
    """
    Preprocess a JSONL file to CSV format for the topic modeling system.
    
    Args:
        input_file: Path to input JSONL file
        output_file: Path to output CSV file
        user_id_field: Field name for user ID in JSONL
        timestamp_field: Field name for timestamp in JSONL
        content_field: Field name for post content in JSONL
        additional_fields: Additional fields to include in output
    """
    # Read JSONL file
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse line: {line[:50]}...")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Rename fields to match expected format
    field_mapping = {
        user_id_field: 'user_id',
        timestamp_field: 'timestamp',
        content_field: 'post_content'
    }
    
    df = df.rename(columns=field_mapping)
    
    # Include additional fields if specified
    if additional_fields:
        fields_to_keep = ['user_id', 'timestamp', 'post_content'] + additional_fields
        df = df[fields_to_keep]
    else:
        df = df[['user_id', 'timestamp', 'post_content']]
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with missing data
    df = df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Preprocessed {len(df)} records to {output_file}")
    return df

def preprocess_csv(input_file: str, output_file: str,
                 user_id_col: str = 'user_id',
                 timestamp_col: str = 'timestamp',
                 content_col: str = 'text',
                 additional_cols: Optional[List[str]] = None):
    """
    Preprocess a CSV file for the topic modeling system.
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
        user_id_col: Column name for user ID
        timestamp_col: Column name for timestamp
        content_col: Column name for post content
        additional_cols: Additional columns to include in output
    """
    # Read CSV file
    df = pd.read_csv(input_file)
    
    # Rename columns to match expected format
    col_mapping = {
        user_id_col: 'user_id',
        timestamp_col: 'timestamp',
        content_col: 'post_content'
    }
    
    df = df.rename(columns=col_mapping)
    
    # Include additional columns if specified
    if additional_cols:
        cols_to_keep = ['user_id', 'timestamp', 'post_content'] + additional_cols
        df = df[cols_to_keep]
    else:
        df = df[['user_id', 'timestamp', 'post_content']]
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Drop rows with missing data
    df = df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Preprocessed {len(df)} records to {output_file}")
    return df

def combine_multiple_files(input_files: List[str], output_file: str,
                         file_type: str = 'auto',
                         id_field: str = 'user_id',
                         timestamp_field: str = 'timestamp',
                         content_field: str = 'text'):
    """
    Combine multiple input files into a single CSV for processing.
    
    Args:
        input_files: List of input file paths
        output_file: Path to output CSV file
        file_type: Type of input files ('csv', 'jsonl', or 'auto' to detect)
        id_field: Field name for user ID
        timestamp_field: Field name for timestamp
        content_field: Field name for post content
    """
    combined_data = []
    
    for file_path in input_files:
        path = Path(file_path)
        
        # Detect file type if 'auto'
        if file_type == 'auto':
            detected_type = path.suffix.lower()
            if detected_type == '.csv':
                current_type = 'csv'
            elif detected_type in ['.jsonl', '.json']:
                current_type = 'jsonl'
            else:
                print(f"Warning: Could not detect type for {file_path}, skipping...")
                continue
        else:
            current_type = file_type
        
        # Process file based on type
        if current_type == 'csv':
            df = pd.read_csv(file_path)
            df = df.rename(columns={
                id_field: 'user_id',
                timestamp_field: 'timestamp',
                content_field: 'post_content'
            })
        elif current_type == 'jsonl':
            data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        # Rename fields
                        item['user_id'] = item.pop(id_field, None)
                        item['timestamp'] = item.pop(timestamp_field, None)
                        item['post_content'] = item.pop(content_field, None)
                        if all(k in item for k in ['user_id', 'timestamp', 'post_content']):
                            data.append(item)
                    except json.JSONDecodeError:
                        continue
            df = pd.DataFrame(data)
        else:
            print(f"Warning: Unsupported file type {current_type} for {file_path}, skipping...")
            continue
        
        # Keep only necessary columns
        df = df[['user_id', 'timestamp', 'post_content']]
        combined_data.append(df)
    
    # Combine all dataframes
    if not combined_data:
        raise ValueError("No valid data files found!")
    
    combined_df = pd.concat(combined_data, ignore_index=True)
    
    # Convert timestamp to datetime
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
    
    # Drop rows with missing data
    combined_df = combined_df.dropna(subset=['user_id', 'timestamp', 'post_content'])
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp')
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"Combined {len(combined_df)} records from {len(input_files)} files to {output_file}")
    return combined_df

def main():
    parser = argparse.ArgumentParser(description='File preprocessing for topic modeling')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # JSONL to CSV
    jsonl_parser = subparsers.add_parser('jsonl', help='Convert JSONL to CSV')
    jsonl_parser.add_argument('input_file', help='Input JSONL file')
    jsonl_parser.add_argument('output_file', help='Output CSV file')
    jsonl_parser.add_argument('--user-id', default='user_id', help='Field name for user ID')
    jsonl_parser.add_argument('--timestamp', default='created_at', help='Field name for timestamp')
    jsonl_parser.add_argument('--content', default='text', help='Field name for post content')
    
    # Preprocess CSV
    csv_parser = subparsers.add_parser('csv', help='Preprocess CSV file')
    csv_parser.add_argument('input_file', help='Input CSV file')
    csv_parser.add_argument('output_file', help='Output CSV file')
    csv_parser.add_argument('--user-id', default='user_id', help='Column name for user ID')
    csv_parser.add_argument('--timestamp', default='timestamp', help='Column name for timestamp')
    csv_parser.add_argument('--content', default='text', help='Column name for post content')
    
    # Combine multiple files
    combine_parser = subparsers.add_parser('combine', help='Combine multiple files')
    combine_parser.add_argument('input_files', nargs='+', help='Input files to combine')
    combine_parser.add_argument('output_file', help='Output CSV file')
    combine_parser.add_argument('--type', choices=['auto', 'csv', 'jsonl'], default='auto', 
                               help='Type of input files')
    combine_parser.add_argument('--user-id', default='user_id', help='Field name for user ID')
    combine_parser.add_argument('--timestamp', default='timestamp', help='Field name for timestamp')
    combine_parser.add_argument('--content', default='text', help='Field name for post content')
    
    args = parser.parse_args()
    
    if args.command == 'jsonl':
        preprocess_jsonl_to_csv(args.input_file, args.output_file, 
                              args.user_id, args.timestamp, args.content)
    
    elif args.command == 'csv':
        preprocess_csv(args.input_file, args.output_file,
                     args.user_id, args.timestamp, args.content)
    
    elif args.command == 'combine':
        combine_multiple_files(args.input_files, args.output_file,
                             args.type, args.user_id, args.timestamp, args.content)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
