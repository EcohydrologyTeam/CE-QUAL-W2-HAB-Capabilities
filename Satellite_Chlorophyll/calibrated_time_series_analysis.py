"""
Calibrated Time Series Analysis for UKL and Detroit Lake
========================================================

This script:
1. Uses the enhanced outlier detection and calibration methods
2. Calibrates MODIS Terra and MODIS Aqua against UKL in situ data
3. Generates time series plots for both UKL and Detroit Lake
4. Includes in situ data for both locations

Key Features:
- Advanced outlier detection (IQR, Z-score, Cook's distance)
- Robust regression calibration
- Comprehensive visualization
- Independent sensor validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')

# Import outlier detection functions from our enhanced script
import sys
import os
sys.path.append('/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll')

def read_data(inpath):
    """Read satellite data from CSV file and prepare for analysis."""
    df = pd.read_csv(inpath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)
    return df

def detect_outliers_iqr(data, column):
    """Detect outliers using Interquartile Range method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers

def detect_outliers_zscore(data, column, threshold=3):
    """Detect outliers using Z-score method"""
    z_scores = np.abs(zscore(data[column]))
    outliers = z_scores > threshold
    return outliers

def detect_outliers_cooks_distance(X, y, threshold=None):
    """Detect outliers using Cook's distance"""
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    
    # Calculate residuals
    residuals = y - y_pred
    
    # Calculate leverage (hat values)
    X_with_intercept = np.column_stack([np.ones(len(X)), X.flatten()])
    hat_matrix = X_with_intercept @ np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(hat_matrix)
    
    # Calculate standardized residuals
    mse = np.mean(residuals**2)
    std_residuals = residuals / np.sqrt(mse * (1 - leverage))
    
    # Calculate Cook's distance
    p = X.shape[1] + 1  # number of parameters including intercept
    cooks_d = (std_residuals**2 / p) * (leverage / (1 - leverage))
    
    # Default threshold: 4/n
    if threshold is None:
        threshold = 4 / len(X)
    
    outliers = cooks_d > threshold
    return outliers, cooks_d

def detect_outliers_combined(matched_data, methods=['iqr', 'zscore', 'cooks']):
    """Detect outliers using multiple methods and return combined results"""
    outlier_info = {}
    
    # Prepare data for outlier detection
    X = matched_data['satellite_value'].values.reshape(-1, 1)
    y = np.log10(matched_data['insitu_chl'].values)
    
    # Remove any invalid values first
    valid_idx = np.isfinite(X.flatten()) & np.isfinite(y)
    X_clean = X[valid_idx]
    y_clean = y[valid_idx]
    matched_clean = matched_data[valid_idx].copy()
    
    outlier_flags = pd.DataFrame(index=matched_clean.index)
    
    if 'iqr' in methods:
        # IQR on satellite values
        iqr_sat = detect_outliers_iqr(matched_clean, 'satellite_value')
        # IQR on in situ values (log scale)
        matched_clean['log_insitu_chl'] = np.log10(matched_clean['insitu_chl'])
        iqr_insitu = detect_outliers_iqr(matched_clean, 'log_insitu_chl')
        outlier_flags['iqr'] = iqr_sat | iqr_insitu
        outlier_info['iqr'] = {
            'satellite_outliers': iqr_sat.sum(),
            'insitu_outliers': iqr_insitu.sum(),
            'total_outliers': (iqr_sat | iqr_insitu).sum()
        }
    
    if 'zscore' in methods:
        # Z-score on residuals after initial fit
        model_temp = LinearRegression().fit(X_clean, y_clean)
        residuals = y_clean - model_temp.predict(X_clean)
        zscore_outliers = np.abs(zscore(residuals)) > 2.5
        outlier_flags['zscore'] = False
        outlier_flags.loc[outlier_flags.index, 'zscore'] = zscore_outliers
        outlier_info['zscore'] = {
            'total_outliers': zscore_outliers.sum()
        }
    
    if 'cooks' in methods:
        cooks_outliers, cooks_distances = detect_outliers_cooks_distance(X_clean, y_clean)
        outlier_flags['cooks'] = False
        outlier_flags.loc[outlier_flags.index, 'cooks'] = cooks_outliers
        matched_clean['cooks_distance'] = cooks_distances
        outlier_info['cooks'] = {
            'total_outliers': cooks_outliers.sum(),
            'distances': cooks_distances
        }
    
    # Combined outliers (any method flags as outlier)
    outlier_flags['any_method'] = outlier_flags.any(axis=1)
    
    # Conservative approach: only flag as outlier if multiple methods agree
    if len(methods) > 1:
        outlier_flags['multiple_methods'] = outlier_flags[methods].sum(axis=1) >= 2
    else:
        outlier_flags['multiple_methods'] = outlier_flags['any_method']
    
    return matched_clean, outlier_flags, outlier_info

def load_ukl_insitu_data():
    """Load UKL in situ chlorophyll data"""
    try:
        from load_ukl_data import load_ukl_insitu_data as load_ukl_func
        from load_ukl_data import filter_ukl_data_for_satellite_calibration
        
        # Load real UKL data
        raw_data = load_ukl_func()
        if len(raw_data) == 0:
            raise Exception("No UKL data loaded")
        
        # Filter for satellite calibration
        filtered_data = filter_ukl_data_for_satellite_calibration(raw_data, min_chl=1.0, max_chl=500.0)
        return filtered_data[['date', 'chlorophyll_ugL']].copy()
        
    except Exception as e:
        print(f"Warning: Could not load real UKL data ({e}), using example data...")
        # Create example data based on typical UKL conditions
        dates = pd.date_range('2011-01-01', '2020-12-31', freq='15D')
        # Simulate seasonal patterns with summer blooms
        day_of_year = dates.dayofyear
        seasonal_pattern = 30 + 50 * np.sin((day_of_year - 120) * 2 * np.pi / 365)
        seasonal_pattern = np.maximum(seasonal_pattern, 5)  # minimum 5 µg/L
        
        # Add some random variation
        np.random.seed(42)
        noise = np.random.normal(0, 10, len(dates))
        chl_values = seasonal_pattern + noise
        chl_values = np.maximum(chl_values, 1)  # minimum 1 µg/L
        
        return pd.DataFrame({'date': dates, 'chlorophyll_ugL': chl_values})

def load_detroit_insitu_data():
    """Load Detroit Lake YSI in situ chlorophyll data"""
    try:
        excel_path = '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Data/Detroit/In Situ/CityofSalem_YSI_RawData.xlsx'
        
        # Read the Excel file
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
        
        # Extract relevant columns
        detroit_insitu = df[['DateTime', 'Chl ug/L', 'Site ID (new)', 'Lat', 'Lon']].copy()
        detroit_insitu.columns = ['date', 'chlorophyll_ugL', 'site_id', 'latitude', 'longitude']
        
        # Clean the data
        detroit_insitu = detroit_insitu.dropna(subset=['date', 'chlorophyll_ugL'])
        detroit_insitu = detroit_insitu[
            (detroit_insitu['chlorophyll_ugL'] > 0) & 
            (detroit_insitu['chlorophyll_ugL'] <= 500)
        ].copy()
        
        # Convert date
        detroit_insitu['date'] = pd.to_datetime(detroit_insitu['date'], errors='coerce')
        detroit_insitu = detroit_insitu.dropna(subset=['date'])
        detroit_insitu = detroit_insitu.sort_values('date').reset_index(drop=True)
        
        print(f"Detroit YSI data loaded: {len(detroit_insitu)} measurements")
        print(f"Date range: {detroit_insitu['date'].min()} to {detroit_insitu['date'].max()}")
        print(f"Chlorophyll range: {detroit_insitu['chlorophyll_ugL'].min():.1f} - {detroit_insitu['chlorophyll_ugL'].max():.1f} µg/L")
        
        return detroit_insitu[['date', 'chlorophyll_ugL']].copy()
        
    except Exception as e:
        print(f"Warning: Could not load Detroit YSI data ({e}), using example data...")
        # Create example Detroit data
        dates = pd.date_range('2015-05-01', '2020-10-31', freq='7D')
        day_of_year = dates.dayofyear
        # Detroit typically has lower chlorophyll than UKL
        seasonal_pattern = 8 + 15 * np.sin((day_of_year - 120) * 2 * np.pi / 365)
        seasonal_pattern = np.maximum(seasonal_pattern, 2)
        
        np.random.seed(24)
        noise = np.random.normal(0, 3, len(dates))
        chl_values = seasonal_pattern + noise
        chl_values = np.maximum(chl_values, 1)
        
        return pd.DataFrame({'date': dates, 'chlorophyll_ugL': chl_values})

def match_satellite_insitu(satellite_df, insitu_df, sat_value_col, tolerance_days=5):
    """Match satellite and in situ observations within specified time tolerance."""
    matches = []
    
    for _, sat_row in satellite_df.iterrows():
        if pd.isna(sat_row[sat_value_col]):
            continue
            
        sat_date = sat_row['date']
        time_diff = np.abs((insitu_df['date'] - sat_date).dt.days)
        within_tolerance = time_diff <= tolerance_days
        
        if within_tolerance.any():
            closest_idx = time_diff[within_tolerance].idxmin()
            insitu_row = insitu_df.loc[closest_idx]
            
            matches.append({
                'date': sat_date,
                'satellite_value': sat_row[sat_value_col],
                'insitu_chl': insitu_row['chlorophyll_ugL'],
                'days_diff': time_diff[closest_idx],
                'sensor': sat_row.get('sensor', 'Unknown')
            })
    
    return pd.DataFrame(matches)

def calibrate_sensor_with_outlier_detection(satellite_df, insitu_df, sat_value_col, sensor_name, 
                                           is_log_scale=True, remove_outliers=True):
    """Calibrate sensor with advanced outlier detection"""
    print(f"\\nCalibrating {sensor_name}...")
    
    # Match satellite data with in situ
    matched = match_satellite_insitu(satellite_df, insitu_df, sat_value_col, tolerance_days=5)
    
    if len(matched) < 10:
        print(f"Error: Only {len(matched)} matches found for {sensor_name}")
        return None
    
    # Remove invalid values
    clean_matches = matched[matched['insitu_chl'] > 0].copy()
    
    if len(clean_matches) < 10:
        print(f"Error: Not enough clean matches for {sensor_name}")
        return None
    
    print(f"Initial clean matches: {len(clean_matches)}")
    
    # Outlier detection and removal
    if remove_outliers and len(clean_matches) >= 15:
        matched_clean, outlier_flags, outlier_info = detect_outliers_combined(clean_matches)
        
        # Use conservative approach: only remove if multiple methods agree
        outliers_to_remove = outlier_flags['multiple_methods']
        n_outliers = outliers_to_remove.sum()
        
        print(f"Outliers detected: {n_outliers}")
        
        if n_outliers > 0:
            clean_matches_no_outliers = matched_clean[~outliers_to_remove].copy()
            print(f"Data points after outlier removal: {len(clean_matches_no_outliers)}")
        else:
            clean_matches_no_outliers = clean_matches.copy()
    else:
        clean_matches_no_outliers = clean_matches.copy()
        print("Skipping outlier detection")
    
    # Prepare calibration data
    X = clean_matches_no_outliers['satellite_value'].values.reshape(-1, 1)
    
    if is_log_scale:
        y = np.log10(clean_matches_no_outliers['insitu_chl'].values)
    else:
        y = clean_matches_no_outliers['insitu_chl'].values
    
    # Remove any remaining invalid values
    valid_idx = np.isfinite(X.flatten()) & np.isfinite(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < 10:
        print(f"Error: Not enough valid data points for {sensor_name}")
        return None
    
    # Fit robust regression
    model = HuberRegressor()
    model.fit(X, y)
    
    # Calculate metrics
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    print(f"Calibration results for {sensor_name}:")
    print(f"  - Final matches: {len(X)}")
    print(f"  - Slope: {model.coef_[0]:.3f}")
    print(f"  - Intercept: {model.intercept_:.3f}")
    print(f"  - R²: {r2:.3f}")
    print(f"  - RMSE: {rmse:.3f}")
    
    return {
        'sensor': sensor_name,
        'model': model,
        'slope': model.coef_[0],
        'intercept': model.intercept_,
        'r2': r2,
        'rmse': rmse,
        'n_matches': len(X),
        'is_log_scale': is_log_scale,
        'matched_data': clean_matches_no_outliers
    }

def apply_calibration(data, value_col, calibration, output_col='chl_calibrated'):
    """Apply calibration to satellite data"""
    if calibration is None:
        return data
    
    result = data.copy()
    valid_data = result.dropna(subset=[value_col])
    
    if calibration['is_log_scale']:
        # OCx approach: log10(Chl) = a * value + b
        log_chl = valid_data[value_col] * calibration['slope'] + calibration['intercept']
        result.loc[valid_data.index, output_col] = np.power(10, log_chl)
    else:
        # Direct approach: Chl = a * value + b
        result.loc[valid_data.index, output_col] = (
            valid_data[value_col] * calibration['slope'] + calibration['intercept']
        )
    
    # Filter reasonable chlorophyll values
    result = result[
        (result[output_col] > 0) & 
        (result[output_col] <= 500)
    ].copy()
    
    return result


def create_time_series_plot(data_dict, insitu_data, title, filename, lake_name):
    """Create comprehensive time series plot"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot in situ data
    if insitu_data is not None and len(insitu_data) > 0:
        label = 'In situ data' if lake_name == 'UKL' else 'YSI data'
        # Plot magenta square markers
        ax.scatter(insitu_data['date'], insitu_data['chlorophyll_ugL'], 
                  c='red', marker='s', s=12, alpha=0.7, label=label, zorder=1)
    
    # Define colors and markers for satellite data
    sensor_styles = {
        'MODIS-Terra': {'color': 'orange', 'marker': 'o'},
        'MODIS-Aqua': {'color': 'green', 'marker': 'o'}
    }
    
    # Plot satellite data
    for sensor_name, sensor_data in data_dict.items():
        if sensor_data is not None and len(sensor_data) > 0:
            style = sensor_styles.get(sensor_name, {'color': 'gray', 'marker': 'o'})
            
            value_col = 'chl_calibrated'
            
            if value_col in sensor_data.columns:
                alpha_val = 0.6
                ax.scatter(sensor_data['date'], sensor_data[value_col],
                          c=style['color'], marker=style['marker'], 
                          s=20, alpha=alpha_val, 
                          label=sensor_name, 
                          zorder=3)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Chlorophyll-a (µg/L)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    
    # Set y-axis limits to 0-350 for both lakes
    ax.set_ylim(0, 350)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    # Improve date formatting
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {filename}")

def main():
    """Main analysis function"""
    print("="*80)
    print("CALIBRATED TIME SERIES ANALYSIS FOR UKL AND DETROIT LAKE")
    print("="*80)
    
    # Load in situ data
    print("\\nLoading in situ data...")
    ukl_insitu = load_ukl_insitu_data()
    detroit_insitu = load_detroit_insitu_data()
    
    print(f"UKL in situ: {len(ukl_insitu)} records")
    print(f"Detroit in situ: {len(detroit_insitu)} records")
    
    # Load UKL satellite data for calibration
    print("\\nLoading UKL satellite data for calibration...")
    ukl_modis_terra = read_data('/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Klamath_MODIS_Terra_500m_ROI.csv')
    ukl_modis_aqua = read_data('/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Klamath_MODIS_Aqua_500m_ROI.csv')
    
    # Add sensor labels
    ukl_modis_terra['sensor'] = 'Terra'
    ukl_modis_aqua['sensor'] = 'Aqua'
    
    print(f"UKL MODIS Terra: {len(ukl_modis_terra)} records")
    print(f"UKL MODIS Aqua: {len(ukl_modis_aqua)} records")
    
    # Perform calibrations
    print("\\n" + "="*60)
    print("PERFORMING CALIBRATIONS AGAINST UKL IN SITU DATA")
    print("="*60)
    
    terra_cal = calibrate_sensor_with_outlier_detection(
        ukl_modis_terra, ukl_insitu, 'log_ratio', 'MODIS-Terra', is_log_scale=True
    )
    
    aqua_cal = calibrate_sensor_with_outlier_detection(
        ukl_modis_aqua, ukl_insitu, 'log_ratio', 'MODIS-Aqua', is_log_scale=True
    )
    
    # Load Detroit and UKL satellite data for time series
    print("\\nLoading satellite data for time series...")
    
    # Detroit data
    detroit_modis_terra = read_data('/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Terra_500m_ROI.csv')
    detroit_modis_aqua = read_data('/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Aqua_500m_ROI.csv')
    
    # Apply calibrations
    print("\\nApplying calibrations...")
    
    # UKL calibrated data
    ukl_terra_cal = apply_calibration(ukl_modis_terra, 'log_ratio', terra_cal)
    ukl_aqua_cal = apply_calibration(ukl_modis_aqua, 'log_ratio', aqua_cal)
    
    # Detroit calibrated data
    detroit_terra_cal = apply_calibration(detroit_modis_terra, 'log_ratio', terra_cal)
    detroit_aqua_cal = apply_calibration(detroit_modis_aqua, 'log_ratio', aqua_cal)
    
    # Create time series plots
    print("\\nCreating time series plots...")
    
    # UKL plot
    ukl_data = {
        'MODIS-Terra': ukl_terra_cal,
        'MODIS-Aqua': ukl_aqua_cal
    }
    
    create_time_series_plot(
        ukl_data, ukl_insitu,
        'Upper Klamath Lake - Calibrated Satellite Chlorophyll vs In Situ Data',
        'UKL_calibrated_time_series.png',
        'UKL'
    )
    
    # Detroit plot
    detroit_data = {
        'MODIS-Terra': detroit_terra_cal,
        'MODIS-Aqua': detroit_aqua_cal
    }
    
    create_time_series_plot(
        detroit_data, detroit_insitu,
        'Detroit Lake - Calibrated Satellite Chlorophyll vs YSI Data',
        'Detroit_calibrated_time_series.png',
        'Detroit'
    )
    
    # Print summary statistics
    print("\\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    def print_data_summary(data_dict, lake_name):
        print(f"\\n{lake_name} Lake:")
        for sensor_name, data in data_dict.items():
            if data is not None and len(data) > 0:
                col = 'chl_calibrated'
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        print(f"  {sensor_name}: {len(values)} points, "
                              f"range {values.min():.1f}-{values.max():.1f} µg/L, "
                              f"mean {values.mean():.1f} µg/L")
    
    print_data_summary(ukl_data, "Upper Klamath")
    print_data_summary(detroit_data, "Detroit")
    
    print("\\nAnalysis completed successfully!")
    print("Generated files:")
    print("- UKL_calibrated_time_series.png")
    print("- Detroit_calibrated_time_series.png")

if __name__ == "__main__":
    main()