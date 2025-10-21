"""
Data labeling module for generating ground-truth labels from ASOS observations.

This module provides fucntions to:
1. Extract timestamps from image filenames.
2. Load and clean ASOS weather station data.
3. Generate prediction targets (classification or regression)
4. Create final labeled datasets for model training
"""

import datetime
from pathlib import Path
import re
import os
import glob

import pandas as pd
import numpy as np
from tqdm import tqdm


def extract_timestamp_from_image(image_path: str) -> datetime.datetime:
    """
    Parses the datetime object from an image filename.

    Arguments:
        image_path: Path to the image file (e.g., './data/7410/snap_7410_20240809_144412_c01d11bc.jpg')
    
    Returns:
        datetime.datetime object parsed from the filename

    print(extract_timestamp_from_image('./data/7410/snap_7410_20240809_144412_c01d11bc.jpg'))
    2024-08-09 14:44:12
    """

    filename = Path(image_path).stem

    match = re.search(r'(\d{8}_\d{6})', filename)
    if not match:
        raise ValueError(f"No valid timestamp found in filename '{filename}'")
    
    timestamp_str = match.group(1)

    try:
        image_datetime = datetime.datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        return image_datetime
    except ValueError as e:
        raise ValueError(f"Could not parse timestamp '{timestamp_str}' in filename '{filename}': {e}")
    

def load_and_clean_asos_data(asos_file_path: str) -> pd.DataFrame:
    """
    Loads the historical ASOS CSV data and cleans precipitation values.

    Arguments:
        asos_file_path: Path to the ASOS CSV file

    Returns: 
        pd.DataFrame with columns: timestamp, precipitation_value

    Note:
        For now, if there is data missing or not reported, it is treated as "no rain" and precipitation will be 0.0.
    """

    asos_df = pd.read_csv(asos_file_path)

    timestamp_cols = ['valid', 'timestamp', 'datetime', 'date', 'time']
    timestamp_col = None

    for col in timestamp_cols:
        if col in asos_df.columns:
            timestamp_col = col
            break

    if timestamp_col is None:
        raise ValueError(f"Could not find timestamp column. Available columns: {asos_df.columns.tolist()}")
    
    asos_df['timestamp'] = pd.to_datetime(asos_df[timestamp_col])

    precip_cols = ['p01i', 'precip', 'precipitation', 'p01m', 'p01']
    precip_col = None
    for col in precip_cols:
        if col in asos_df.columns:
            precip_col = col
            break
    
    if precip_col is None:
        raise ValueError(f"Could not find precipitation column. Available columns: {asos_df.columns.tolist()}")

    asos_df[precip_col] = asos_df[precip_col].replace({
        'M': np.nan,
        'T': 0.0,
        '': np.nan,
        ' ': np.nan
    })

    asos_df['precipitation_value'] = pd.to_numeric(asos_df[precip_col], errors='coerce')
    asos_df['precipitation_value'] = asos_df['precipitation_value'].fillna(0.0)

    asos_df = asos_df.sort_values('timestamp').reset_index(drop=True)

    return asos_df[['timestamp', 'precipitation_value']]


def generate_prediction_target(
        image_datetime: datetime.datetime,
        asos_df: pd.DataFrame,
        lead_time_min: int,
        mode:str,
        precip_threshold:float
) -> float:
    """
    Determines the precipitation target by querying ASOS data.

    Arguments:
        image_datetime: Timestamp of the sky image
        asos_df: DataFrame with ASOS observations
        lead_time_min: Lead time in minutes
        mode: 'binary' for classification or 'regression' for continuous values
        precip_threshold: Threshold to define storm in binary mode

    Returns:
        target_value: int for binary mode, float for regression mode
    """

    start_time = image_datetime + datetime.timedelta(minutes=lead_time_min)
    end_time = start_time + datetime.timedelta(minutes=60)

    if start_time.tzinfo is not None:
        start_time = start_time.replace(tzinfo=None)
    if end_time.tzinfo is not None:
        end_time = end_time.replace(tzinfo=None)
    
    try:
        mask = (asos_df['timestamp'] >= start_time) & (asos_df['timestamp'] < end_time)
        window_data = asos_df.loc[mask]
    except Exception as e:
        print(f"Error filtering data for time window {start_time} to {end_time}: {e}")
        return np.nan

    if window_data.empty:
        return np.nan
    
    total_precip = window_data['precipitation_value'].sum()

    if mode == 'binary':
        return int(total_precip >= precip_threshold)
    elif mode == 'regression':
        return float(total_precip)
    else:
        raise ValueError(f"Mode must be 'binary' or 'regression', got {mode}")
    

def create_final_label_dataset(
        image_dir: str,
        asos_df: pd.DataFrame,
        lead_time_min: int,
        mode: str,
        precip_threshold: float
) -> pd.DataFrame:
    """
    Creates the final labeled dataset by processinf all images.

    Arguments:
        image_dir: Directory containing sky images
        asos_df: DataFrame with ASOS observations
        lead_time_min: Lead time in minutes for prediction
        mode: 'binary' or 'regression'
        precip_threshold: Threshold for binary classification
    
    Returns:
        pd.DataFrame with columns: image_path, target_value
    """

    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))

    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    print(f"Found {len(image_paths)} images in {image_dir}")

    valid_image_paths = []
    target_values = []

    for image_path in tqdm(image_paths, desc="Labeling images"):
        try:
            image_datetime = extract_timestamp_from_image(image_path)
            
            target_value = generate_prediction_target(
                image_datetime,
                asos_df,
                lead_time_min,
                mode,
                precip_threshold
            )
            
            if not np.isnan(target_value):
                valid_image_paths.append(image_path)
                target_values.append(target_value)
        
        except Exception as e:
            print(f"Warning: Skipping {image_path} due to error: {e}")
            continue
    
    labeled_data_df = pd.DataFrame({
        'image_path': valid_image_paths,
        'target_value': target_values
    })
    
    print(f"\nLabeling Summary:")
    print(f"  Total valid samples: {len(labeled_data_df)}")
    
    if mode == 'binary':
        storm_count = (labeled_data_df['target_value'] == 1).sum()
        no_storm_count = (labeled_data_df['target_value'] == 0).sum()
        print(f"  Storm samples (1): {storm_count} ({storm_count/len(labeled_data_df)*100:.1f}%)")
        print(f"  No storm samples (0): {no_storm_count} ({no_storm_count/len(labeled_data_df)*100:.1f}%)")
    else:
        print(f"  Mean precipitation: {labeled_data_df['target_value'].mean():.4f} inches")
        print(f"  Max precipitation: {labeled_data_df['target_value'].max():.4f} inches")
        print(f"  Samples with precipitation > 0: {(labeled_data_df['target_value'] > 0).sum()}")
    
    return labeled_data_df