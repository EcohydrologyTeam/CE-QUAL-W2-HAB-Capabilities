"""
Multi-Method Calibrated Time Series Analysis with CSV Export for UKL and Detroit Lake
======================================================================================

This script performs calibration for three different MODIS chlorophyll extraction methods:
1. Green/Red ratio (500m) - Traditional ocean color approach
2. NIR/Red ratio (500m) - Optimized for turbid Case 2 waters
3. NIR/Red ratio (250m) - High spatial resolution 8-day composites

Key Features:
- Advanced outlier detection (IQR, Z-score, Cook's distance)
- Robust regression calibration
- Comprehensive visualization for all three methods
- Independent sensor validation
- CSV export functionality for calibrated data
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
    
    # Check if datetime column exists, otherwise use date column
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        # Also keep date for backward compatibility
        df['date'] = pd.to_datetime(df['datetime'].dt.date)
        df.sort_values('datetime', inplace=True)
    else:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        # Create a datetime column with default time of 00:00:00
        df['datetime'] = df['date']
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
            
        # Use datetime if available, otherwise fall back to date
        sat_datetime = sat_row.get('datetime', sat_row['date'])
        if pd.isna(sat_datetime):
            sat_datetime = sat_row['date']
        sat_date = pd.to_datetime(sat_datetime).normalize() if not pd.isna(sat_datetime) else sat_row['date']
        
        time_diff = np.abs((insitu_df['date'] - sat_date).dt.days)
        within_tolerance = time_diff <= tolerance_days
        
        if within_tolerance.any():
            closest_idx = time_diff[within_tolerance].idxmin()
            insitu_row = insitu_df.loc[closest_idx]
            
            matches.append({
                'date': sat_date,
                'datetime': sat_datetime,
                'satellite_value': sat_row[sat_value_col],
                'insitu_chl': insitu_row['chlorophyll_ugL'],
                'days_diff': time_diff[closest_idx],
                'sensor': sat_row.get('sensor', 'Unknown')
            })
    
    return pd.DataFrame(matches)

def calibrate_sensor_with_outlier_detection(satellite_df, insitu_df, sat_value_col, sensor_name, 
                                           is_log_scale=True, remove_outliers=True):
    """Calibrate sensor with advanced outlier detection"""
    print(f"\nCalibrating {sensor_name}...")
    
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
    
    # Initialize outlier results
    outlier_results = {'n_outliers_removed': 0}
    
    # Outlier detection and removal
    if remove_outliers and len(clean_matches) >= 15:
        matched_clean, outlier_flags, outlier_info = detect_outliers_combined(clean_matches)
        
        # Use conservative approach: only remove if multiple methods agree
        outliers_to_remove = outlier_flags['multiple_methods']
        n_outliers = outliers_to_remove.sum()
        
        print(f"Outliers detected: {n_outliers}")
        
        if n_outliers > 0:
            clean_matches_no_outliers = matched_clean[~outliers_to_remove].copy()
            outlier_data = matched_clean[outliers_to_remove].copy()
            print(f"Data points after outlier removal: {len(clean_matches_no_outliers)}")
            
            outlier_results = {
                'original_data': clean_matches,
                'outlier_flags': outlier_flags,
                'outlier_info': outlier_info,
                'outliers_removed': outlier_data,
                'n_outliers_removed': n_outliers
            }
        else:
            clean_matches_no_outliers = clean_matches.copy()
            print("No outliers detected by multiple methods")
            outlier_results['n_outliers_removed'] = 0
    else:
        clean_matches_no_outliers = clean_matches.copy()
        print("Skipping outlier detection")
    
    # Prepare calibration data for both original and cleaned data
    X_orig = clean_matches['satellite_value'].values.reshape(-1, 1)
    if is_log_scale:
        y_orig = np.log10(clean_matches['insitu_chl'].values)
    else:
        y_orig = clean_matches['insitu_chl'].values
    
    # Remove any invalid values from original data
    valid_idx_orig = np.isfinite(X_orig.flatten()) & np.isfinite(y_orig)
    X_orig = X_orig[valid_idx_orig]
    y_orig = y_orig[valid_idx_orig]
    
    # Prepare cleaned calibration data
    X_clean = clean_matches_no_outliers['satellite_value'].values.reshape(-1, 1)
    if is_log_scale:
        y_clean = np.log10(clean_matches_no_outliers['insitu_chl'].values)
    else:
        y_clean = clean_matches_no_outliers['insitu_chl'].values
    
    # Remove any remaining invalid values from cleaned data
    valid_idx_clean = np.isfinite(X_clean.flatten()) & np.isfinite(y_clean)
    X_clean = X_clean[valid_idx_clean]
    y_clean = y_clean[valid_idx_clean]
    
    if len(X_clean) < 10:
        print(f"Error: Not enough valid data points for {sensor_name}")
        return None
    
    # Fit robust regression for both datasets
    model_orig = HuberRegressor()
    model_orig.fit(X_orig, y_orig)
    
    model_clean = HuberRegressor()
    model_clean.fit(X_clean, y_clean)
    
    # Calculate metrics for both models
    y_pred_orig = model_orig.predict(X_orig)
    r2_orig = r2_score(y_orig, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_orig, y_pred_orig))
    
    y_pred_clean = model_clean.predict(X_clean)
    r2_clean = r2_score(y_clean, y_pred_clean)
    rmse_clean = np.sqrt(mean_squared_error(y_clean, y_pred_clean))
    
    print(f"Calibration results for {sensor_name}:")
    print(f"  Original data: {len(X_orig)} matches, R² = {r2_orig:.3f}, RMSE = {rmse_orig:.3f}")
    print(f"  Cleaned data: {len(X_clean)} matches, R² = {r2_clean:.3f}, RMSE = {rmse_clean:.3f}")
    print(f"  - Slope: {model_clean.coef_[0]:.3f}")
    print(f"  - Intercept: {model_clean.intercept_:.3f}")
    
    # Create outlier analysis plot if outliers were detected
    if outlier_results['n_outliers_removed'] > 0:
        # Extract method name from sensor name for plotting
        method_name = sensor_name.split(' ', 1)[-1] if ' ' in sensor_name else sensor_name
        sensor_short = sensor_name.split(' ')[0] if ' ' in sensor_name else sensor_name
        
        create_outlier_comparison_plot(
            X_orig, y_orig, X_clean, y_clean, 
            model_orig, model_clean,
            sensor_short, method_name, outlier_results,
            r2_orig, rmse_orig, r2_clean, rmse_clean
        )
    
    return {
        'sensor': sensor_name,
        'model': model_clean,
        'slope': model_clean.coef_[0],
        'intercept': model_clean.intercept_,
        'r2': r2_clean,
        'rmse': rmse_clean,
        'n_matches': len(X_clean),
        'is_log_scale': is_log_scale,
        'matched_data': clean_matches_no_outliers,
        'outlier_results': outlier_results
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

def combine_modis_sensors(terra_data, aqua_data, date_col='date', value_col='chl_calibrated'):
    """Combine MODIS Terra and Aqua data into a single dataset"""
    if terra_data is None and aqua_data is None:
        return None
    
    # Prepare dataframes with sensor labels
    combined_data = []
    
    if terra_data is not None and len(terra_data) > 0:
        # Include datetime if available
        cols = [date_col, value_col]
        if 'datetime' in terra_data.columns:
            cols.append('datetime')
        terra_subset = terra_data[cols].copy()
        terra_subset['sensor'] = 'MODIS-Terra'
        combined_data.append(terra_subset)
    
    if aqua_data is not None and len(aqua_data) > 0:
        # Include datetime if available
        cols = [date_col, value_col]
        if 'datetime' in aqua_data.columns:
            cols.append('datetime')
        aqua_subset = aqua_data[cols].copy()
        aqua_subset['sensor'] = 'MODIS-Aqua'
        combined_data.append(aqua_subset)
    
    if not combined_data:
        return None
    
    # Combine and sort by datetime (or date if datetime not available)
    combined_df = pd.concat(combined_data, ignore_index=True)
    sort_col = 'datetime' if 'datetime' in combined_df.columns else date_col
    combined_df = combined_df.sort_values(sort_col).reset_index(drop=True)
    
    return combined_df

def export_calibrated_data_to_csv(data_dict, lake_name, method_name, output_dir='/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/'):
    """Export calibrated MODIS data to CSV files for a specific method"""
    
    print(f"\nExporting {lake_name} {method_name} calibrated data to CSV files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create method-specific suffix
    method_suffix = method_name.replace(' ', '_').replace('/', '_')
    
    # Export individual sensor files
    for sensor_name, data in data_dict.items():
        if data is not None and len(data) > 0:
            # Prepare data for export with datetime if available
            export_cols = ['date', 'chl_calibrated']
            if 'datetime' in data.columns:
                export_cols = ['datetime', 'date', 'chl_calibrated']
            export_data = data[export_cols].copy()
            
            # Rename columns
            rename_dict = {'chl_calibrated': 'chlorophyll_ugL'}
            export_data = export_data.rename(columns=rename_dict)
            export_data['sensor'] = sensor_name
            export_data['method'] = method_name
            
            # Create filename
            sensor_clean = sensor_name.replace('-', '_').replace(' ', '_')
            filename = f"{lake_name}_{sensor_clean}_{method_suffix}_calibrated_chlorophyll.csv"
            filepath = os.path.join(output_dir, filename)
            
            # Export to CSV
            export_data.to_csv(filepath, index=False)
            print(f"  Exported {sensor_name}: {len(export_data)} records -> {filename}")
    
    # Create combined MODIS file
    terra_data = data_dict.get('MODIS-Terra')
    aqua_data = data_dict.get('MODIS-Aqua')
    combined_data = combine_modis_sensors(terra_data, aqua_data)
    
    if combined_data is not None and len(combined_data) > 0:
        # Prepare combined data for export with datetime if available
        export_cols = ['date', 'chl_calibrated', 'sensor']
        if 'datetime' in combined_data.columns:
            export_cols = ['datetime', 'date', 'chl_calibrated', 'sensor']
        export_combined = combined_data[export_cols].copy()
        
        # Rename columns
        rename_dict = {'chl_calibrated': 'chlorophyll_ugL'}
        export_combined = export_combined.rename(columns=rename_dict)
        export_combined['method'] = method_name
        
        # Create filename for combined data
        combined_filename = f"{lake_name}_MODIS_Combined_{method_suffix}_calibrated_chlorophyll.csv"
        combined_filepath = os.path.join(output_dir, combined_filename)
        
        # Export combined data
        export_combined.to_csv(combined_filepath, index=False)
        print(f"  Exported MODIS Combined: {len(export_combined)} records -> {combined_filename}")

def create_outlier_comparison_plot(X_orig, y_orig, X_clean, y_clean, model_orig, model_clean, 
                                 sensor_name, method_name, outlier_results, r2_orig, rmse_orig, r2_clean, rmse_clean):
    """Create comprehensive plots showing before/after outlier removal"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MODIS {sensor_name} {method_name} Calibration: Outlier Analysis', fontsize=16, fontweight='bold')
    
    # Define consistent axis ranges
    x_min = min(X_orig.min(), X_clean.min()) - 0.1
    x_max = max(X_orig.max(), X_clean.max()) + 0.1
    y_min_log = min(y_orig.min(), y_clean.min()) - 0.1
    y_max_log = max(y_orig.max(), y_clean.max()) + 0.1
    
    # Row 1: Original data
    # Original scatter plot
    axes[0, 0].scatter(X_orig.flatten(), np.power(10, y_orig), alpha=0.6, s=30, color='blue', label='Data points')
    if 'outliers_removed' in outlier_results and len(outlier_results['outliers_removed']) > 0:
        outlier_data = outlier_results['outliers_removed']
        X_outliers = outlier_data['satellite_value'].values
        y_outliers = np.log10(outlier_data['insitu_chl'].values)
        axes[0, 0].scatter(X_outliers, np.power(10, y_outliers), alpha=0.8, s=50, color='red', 
                          marker='x', label=f'Outliers ({len(outlier_data)})')
    
    x_range = np.linspace(x_min, x_max, 100)
    y_range_pred_orig = model_orig.predict(x_range.reshape(-1, 1))
    axes[0, 0].plot(x_range, np.power(10, y_range_pred_orig), 'r-', linewidth=2, label='Regression line')
    axes[0, 0].set_xlabel('Satellite Index')
    axes[0, 0].set_ylabel('In Situ Chlorophyll (µg/L)')
    axes[0, 0].set_title(f'Original Data\nR² = {r2_orig:.3f}, RMSE = {rmse_orig:.3f}')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Original residuals
    residuals_orig = y_orig - model_orig.predict(X_orig)
    axes[0, 1].scatter(model_orig.predict(X_orig), residuals_orig, alpha=0.6, s=30, color='blue')
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted log10(Chl)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Original Data Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Q-Q plot for original residuals
    from scipy import stats
    stats.probplot(residuals_orig, dist="norm", plot=axes[0, 2])
    axes[0, 2].set_title('Original Data Q-Q Plot')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Row 2: Cleaned data
    # Cleaned scatter plot
    axes[1, 0].scatter(X_clean.flatten(), np.power(10, y_clean), alpha=0.6, s=30, color='green', label='Clean data')
    y_range_pred_clean = model_clean.predict(x_range.reshape(-1, 1))
    axes[1, 0].plot(x_range, np.power(10, y_range_pred_clean), 'r-', linewidth=2, label='Regression line')
    axes[1, 0].set_xlabel('Satellite Index')
    axes[1, 0].set_ylabel('In Situ Chlorophyll (µg/L)')
    axes[1, 0].set_title(f'Cleaned Data (Outliers Removed)\nR² = {r2_clean:.3f}, RMSE = {rmse_clean:.3f}')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Cleaned residuals
    residuals_clean = y_clean - model_clean.predict(X_clean)
    axes[1, 1].scatter(model_clean.predict(X_clean), residuals_clean, alpha=0.6, s=30, color='green')
    axes[1, 1].axhline(y=0, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Predicted log10(Chl)')
    axes[1, 1].set_ylabel('Residuals')
    axes[1, 1].set_title('Cleaned Data Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Q-Q plot for cleaned residuals
    stats.probplot(residuals_clean, dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Cleaned Data Q-Q Plot')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Clean method name for filename
    method_clean = method_name.replace(' ', '_').replace('/', '_')
    filename = f'MODIS_{sensor_name}_{method_clean}_outlier_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Outlier analysis plot saved: {filename}")

def create_time_series_plot(data_dict, insitu_data, title, filename, lake_name):
    """Create comprehensive time series plot"""
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot in situ data
    if insitu_data is not None and len(insitu_data) > 0:
        label = 'In situ data' if lake_name == 'UKL' else 'YSI data'
        # Plot red square markers
        ax.scatter(insitu_data['date'], insitu_data['chlorophyll_ugL'], 
                  c='red', marker='s', s=12, alpha=0.7, label=label, zorder=1)
    
    # Define colors and markers for satellite data
    sensor_styles = {
        'MODIS-Terra': {'color': 'orange', 'marker': 'o'},
        'MODIS-Aqua': {'color': 'green', 'marker': 'o'},
        'MODIS-Aqua-250m': {'color': 'blue', 'marker': '^'}
    }
    
    # Plot satellite data using datetime if available
    for sensor_name, sensor_data in data_dict.items():
        if sensor_data is not None and len(sensor_data) > 0:
            style = sensor_styles.get(sensor_name, {'color': 'gray', 'marker': 'o'})
            
            value_col = 'chl_calibrated'
            
            if value_col in sensor_data.columns:
                alpha_val = 0.6
                # Use datetime for x-axis if available, otherwise use date
                x_col = 'datetime' if 'datetime' in sensor_data.columns else 'date'
                ax.scatter(sensor_data[x_col], sensor_data[value_col],
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

def process_method(method_config, ukl_insitu, detroit_insitu):
    """Process a single method configuration"""
    
    method_name = method_config['name']
    print(f"\n{'='*80}")
    print(f"PROCESSING {method_name.upper()} METHOD")
    print(f"{'='*80}")
    
    # Load UKL satellite data for calibration
    print(f"\nLoading UKL satellite data for {method_name}...")
    ukl_modis_terra = None
    ukl_modis_aqua = None
    
    # Check if Terra file exists and load
    if 'ukl_terra_file' in method_config and os.path.exists(method_config['ukl_terra_file']):
        ukl_modis_terra = read_data(method_config['ukl_terra_file'])
        ukl_modis_terra['sensor'] = 'Terra'
        print(f"UKL MODIS Terra: {len(ukl_modis_terra)} records")
    
    # Check if Aqua file exists and load
    if 'ukl_aqua_file' in method_config and os.path.exists(method_config['ukl_aqua_file']):
        ukl_modis_aqua = read_data(method_config['ukl_aqua_file'])
        ukl_modis_aqua['sensor'] = 'Aqua'
        print(f"UKL MODIS Aqua: {len(ukl_modis_aqua)} records")
    
    # Perform calibrations
    print(f"\nPerforming calibrations against UKL in situ data...")
    
    terra_cal = None
    aqua_cal = None
    
    if ukl_modis_terra is not None:
        terra_cal = calibrate_sensor_with_outlier_detection(
            ukl_modis_terra, ukl_insitu, method_config['value_col'], 
            f'MODIS-Terra {method_name}', is_log_scale=True
        )
    
    if ukl_modis_aqua is not None:
        aqua_cal = calibrate_sensor_with_outlier_detection(
            ukl_modis_aqua, ukl_insitu, method_config['value_col'], 
            f'MODIS-Aqua {method_name}', is_log_scale=True
        )
    
    # Load Detroit satellite data for time series
    print(f"\nLoading Detroit satellite data for {method_name}...")
    detroit_modis_terra = None
    detroit_modis_aqua = None
    
    # Check if Detroit Terra file exists and load
    if 'detroit_terra_file' in method_config and os.path.exists(method_config['detroit_terra_file']):
        detroit_modis_terra = read_data(method_config['detroit_terra_file'])
        print(f"Detroit MODIS Terra: {len(detroit_modis_terra)} records")
    
    # Check if Detroit Aqua file exists and load
    if 'detroit_aqua_file' in method_config and os.path.exists(method_config['detroit_aqua_file']):
        detroit_modis_aqua = read_data(method_config['detroit_aqua_file'])
        print(f"Detroit MODIS Aqua: {len(detroit_modis_aqua)} records")
    
    # Apply calibrations
    print(f"\nApplying calibrations for {method_name}...")
    
    # UKL calibrated data
    ukl_terra_cal_data = None
    ukl_aqua_cal_data = None
    
    if ukl_modis_terra is not None and terra_cal is not None:
        ukl_terra_cal_data = apply_calibration(ukl_modis_terra, method_config['value_col'], terra_cal)
    
    if ukl_modis_aqua is not None and aqua_cal is not None:
        ukl_aqua_cal_data = apply_calibration(ukl_modis_aqua, method_config['value_col'], aqua_cal)
    
    # Detroit calibrated data
    detroit_terra_cal_data = None
    detroit_aqua_cal_data = None
    
    if detroit_modis_terra is not None and terra_cal is not None:
        detroit_terra_cal_data = apply_calibration(detroit_modis_terra, method_config['value_col'], terra_cal)
    
    if detroit_modis_aqua is not None and aqua_cal is not None:
        detroit_aqua_cal_data = apply_calibration(detroit_modis_aqua, method_config['value_col'], aqua_cal)
    
    # Prepare data dictionaries for CSV export
    ukl_data = {}
    detroit_data = {}
    
    # Add Terra data if exists
    if ukl_terra_cal_data is not None:
        ukl_data['MODIS-Terra'] = ukl_terra_cal_data
    if detroit_terra_cal_data is not None:
        detroit_data['MODIS-Terra'] = detroit_terra_cal_data
    
    # Add Aqua data if exists - use special label for 250m
    if ukl_aqua_cal_data is not None:
        sensor_label = 'MODIS-Aqua-250m' if '250m' in method_name else 'MODIS-Aqua'
        ukl_data[sensor_label] = ukl_aqua_cal_data
    if detroit_aqua_cal_data is not None:
        sensor_label = 'MODIS-Aqua-250m' if '250m' in method_name else 'MODIS-Aqua'
        detroit_data[sensor_label] = detroit_aqua_cal_data
    
    # Export calibrated data to CSV files
    export_calibrated_data_to_csv(ukl_data, "UKL", method_name)
    export_calibrated_data_to_csv(detroit_data, "Detroit", method_name)
    
    # Create time series plots
    print(f"\nCreating time series plots for {method_name}...")
    
    # UKL plot
    create_time_series_plot(
        ukl_data, ukl_insitu,
        f'Upper Klamath Lake - Calibrated Satellite Chlorophyll vs In Situ Data ({method_name})',
        f'UKL_calibrated_time_series_{method_name.replace(" ", "_").replace("/", "_")}.png',
        'UKL'
    )
    
    # Detroit plot
    create_time_series_plot(
        detroit_data, detroit_insitu,
        f'Detroit Lake - Calibrated Satellite Chlorophyll vs YSI Data ({method_name})',
        f'Detroit_calibrated_time_series_{method_name.replace(" ", "_").replace("/", "_")}.png',
        'Detroit'
    )
    
    return ukl_data, detroit_data

def main():
    """Main analysis function"""
    print("="*80)
    print("MULTI-METHOD CALIBRATED TIME SERIES ANALYSIS")
    print("="*80)
    
    # Load in situ data
    print("\nLoading in situ data...")
    ukl_insitu = load_ukl_insitu_data()
    detroit_insitu = load_detroit_insitu_data()
    
    print(f"UKL in situ: {len(ukl_insitu)} records")
    print(f"Detroit in situ: {len(detroit_insitu)} records")
    
    # Define method configurations
    methods = [
        {
            'name': 'Green/Red 500m',
            'ukl_terra_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Klamath_MODIS_Terra_Chlorophyll_B4_Green_B1_Red_500m.csv',
            'ukl_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Klamath_MODIS_Aqua_Chlorophyll_B4_Green_B1_Red_500m.csv',
            'detroit_terra_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Terra_Chlorophyll_B4_Green_B1_Red_500m.csv',
            'detroit_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Aqua_Chlorophyll_B4_Green_B1_Red_500m.csv',
            'value_col': 'log_ratio'
        },
        {
            'name': 'NIR/Red 500m',
            'ukl_terra_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/UKL_MODIS_Terra_Chlorophyll_B2_NIR_B1_Red_500m.csv',
            'ukl_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/UKL_MODIS_Aqua_Chlorophyll_B2_NIR_B1_Red_500m.csv',
            'detroit_terra_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Terra_Chlorophyll_B2_NIR_B1_Red_500m.csv',
            'detroit_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Aqua_Chlorophyll_B2_NIR_B1_Red_500m.csv',
            'value_col': 'log_nir_red'
        },
        {
            'name': 'NIR/Red 250m',
            'ukl_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/UKL_MODIS_Aqua_Chlorophyll_B2_NIR_B1_Red_250m.csv',
            'detroit_aqua_file': '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Satellite_Chlorophyll/Detroit_MODIS_Aqua_Chlorophyll_B2_NIR_B1_Red_250m.csv',
            'value_col': 'log_nir_red'
        }
    ]
    
    # Process each method
    all_results = {}
    for method_config in methods:
        ukl_data, detroit_data = process_method(method_config, ukl_insitu, detroit_insitu)
        all_results[method_config['name']] = {
            'ukl': ukl_data,
            'detroit': detroit_data
        }
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS FOR ALL METHODS")
    print("="*80)
    
    for method_name, results in all_results.items():
        print(f"\n{method_name}:")
        print("-" * 40)
        
        # UKL statistics
        print("Upper Klamath Lake:")
        for sensor_name, data in results['ukl'].items():
            if data is not None and len(data) > 0:
                col = 'chl_calibrated'
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        print(f"  {sensor_name}: {len(values)} points, "
                              f"range {values.min():.1f}-{values.max():.1f} µg/L, "
                              f"mean {values.mean():.1f} µg/L")
        
        # Detroit statistics
        print("Detroit Lake:")
        for sensor_name, data in results['detroit'].items():
            if data is not None and len(data) > 0:
                col = 'chl_calibrated'
                if col in data.columns:
                    values = data[col].dropna()
                    if len(values) > 0:
                        print(f"  {sensor_name}: {len(values)} points, "
                              f"range {values.min():.1f}-{values.max():.1f} µg/L, "
                              f"mean {values.mean():.1f} µg/L")
    
    print("\n" + "="*80)
    print("Analysis completed successfully!")
    print("Generated files for each method:")
    print("- Time series plots")
    print("- Individual sensor CSV files")
    print("- Combined MODIS CSV files")

if __name__ == "__main__":
    main()