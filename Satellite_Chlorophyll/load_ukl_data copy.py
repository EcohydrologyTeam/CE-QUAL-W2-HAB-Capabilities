"""
Load and process Upper Klamath Lake in situ chlorophyll data

This script loads the UKL in situ chlorophyll measurements from the Klamath Tribes
and CEDEN monitoring programs.
"""

import pandas as pd
import numpy as np
from datetime import datetime

def load_ukl_insitu_data(csv_path='/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Data/Upper_Klamath_Lake/Klamath K.csv'):
    """
    Load UKL in situ chlorophyll data from WQP export
    
    Returns:
        DataFrame with columns: date, chlorophyll_ugL, organization, location
    """
    print(f"Loading UKL in situ data from: {csv_path}")
    
    # Read the CSV file
    df = pd.read_csv(csv_path, encoding='utf-8-sig')  # Handle BOM
    
    # Filter for chlorophyll measurements
    chl_data = df[df['CharacteristicName'] == 'Chlorophyll a'].copy()
    
    print(f"Found {len(chl_data)} chlorophyll measurements")
    
    # Extract relevant columns
    processed_data = []
    
    for _, row in chl_data.iterrows():
        try:
            # Parse date from ActivityStartDate 
            date_str = str(row['ActivityStartDate'])
            
            # Handle Excel date numbers (e.g., 40651)
            if date_str.isdigit() and len(date_str) == 5:
                # Excel date (days since 1900-01-01, but Excel counts from 1900-01-00)
                excel_date = int(date_str)
                # Convert Excel date to datetime (subtract 1 because Excel counts from 1900-01-00)
                date = datetime(1899, 12, 30) + pd.Timedelta(days=excel_date)
            else:
                # Try to parse as regular date string
                date = pd.to_datetime(date_str, errors='coerce')
            
            if pd.isna(date):
                continue
                
            # Get chlorophyll value
            chl_value = row['ResultMeasureValue']
            if pd.isna(chl_value) or chl_value <= 0:
                continue
                
            # Get metadata
            org = row['OrganizationIdentifier']
            location = row['MonitoringLocationName']
            
            processed_data.append({
                'date': date,
                'chlorophyll_ugL': float(chl_value),
                'organization': org,
                'location': location,
                'latitude': row.get('ActivityLocation/LatitudeMeasure', np.nan),
                'longitude': row.get('ActivityLocation/LongitudeMeasure', np.nan)
            })
            
        except Exception as e:
            print(f"Error processing row: {e}")
            continue
    
    if not processed_data:
        print("ERROR: No valid chlorophyll data found!")
        return pd.DataFrame()
    
    # Create DataFrame
    insitu_df = pd.DataFrame(processed_data)
    
    # Sort by date
    insitu_df = insitu_df.sort_values('date').reset_index(drop=True)
    
    # Basic statistics
    print(f"\nData Summary:")
    print(f"Date range: {insitu_df['date'].min()} to {insitu_df['date'].max()}")
    print(f"Chlorophyll range: {insitu_df['chlorophyll_ugL'].min():.1f} - {insitu_df['chlorophyll_ugL'].max():.1f} µg/L")
    print(f"Mean chlorophyll: {insitu_df['chlorophyll_ugL'].mean():.1f} µg/L")
    print(f"Median chlorophyll: {insitu_df['chlorophyll_ugL'].median():.1f} µg/L")
    print(f"Organizations: {insitu_df['organization'].unique()}")
    print(f"Number of unique locations: {insitu_df['location'].nunique()}")
    
    return insitu_df

def filter_ukl_data_for_satellite_calibration(insitu_df, min_chl=1.0, max_chl=500.0):
    """
    Filter UKL data for satellite calibration
    
    Args:
        insitu_df: DataFrame with UKL in situ data
        min_chl: Minimum chlorophyll threshold
        max_chl: Maximum chlorophyll threshold
    
    Returns:
        Filtered DataFrame
    """
    initial_count = len(insitu_df)
    
    # Filter by chlorophyll range
    filtered = insitu_df[
        (insitu_df['chlorophyll_ugL'] >= min_chl) & 
        (insitu_df['chlorophyll_ugL'] <= max_chl)
    ].copy()
    
    final_count = len(filtered)
    removed_count = initial_count - final_count
    
    print(f"Filtering results:")
    print(f"  Initial records: {initial_count}")
    print(f"  Final records: {final_count}")
    print(f"  Removed: {removed_count} ({100*removed_count/initial_count:.1f}%)")
    
    return filtered

if __name__ == "__main__":
    # Test the function
    data = load_ukl_insitu_data()
    
    if len(data) > 0:
        # Show sample data
        print(f"\nFirst 5 records:")
        print(data.head())
        
        # Filter for satellite calibration
        filtered_data = filter_ukl_data_for_satellite_calibration(data)
        
        # Export to CSV
        output_file = "UKL_insitu_chlorophyll.csv"
        filtered_data.to_csv(output_file, index=False)
        print(f"\nFiltered data exported to: {output_file}")
    else:
        print("No data loaded!")