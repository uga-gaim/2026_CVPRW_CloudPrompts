"""
Data labeling module for generating ground-truth labels from ASOS observations.

This module provides fucntions to:
1. Extract timestamps from image filenames.
2. 
3. 
4. 
"""

import datetime
from pathlib import Path
import re

import pandas as pd
import numpy as np

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
