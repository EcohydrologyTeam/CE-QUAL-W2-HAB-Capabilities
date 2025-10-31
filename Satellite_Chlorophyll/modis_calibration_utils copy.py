"""
MODIS Calibration Utilities
============================
Combines best features from multiple scripts:
- Time series plotting from Calibrate_MODIS_Multi_Method.py
- Outlier detection from test_modis_calibration_outlier_detection.py
- Modular design for notebook integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import zscore
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# OUTLIER DETECTION FUNCTIONS (from test_modis_calibration_outlier_detection.py)
# ==============================================================================

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


# ==============================================================================
# CALIBRATION FUNCTIONS
# ==============================================================================

def calibrate_with_outlier_detection(matched_data, sensor_name, value_col='satellite_value', 
                                     is_log_scale=True, remove_outliers=True):
    """
    Calibrate sensor with advanced outlier detection
    Returns both original and cleaned calibration results
    """
    print(f"\nCalibrating {sensor_name}...")
    
    # Remove invalid values
    clean_matches = matched_data[matched_data['insitu_chl'] > 0].copy()
    
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
    else:
        clean_matches_no_outliers = clean_matches.copy()
        print("Skipping outlier detection")
    
    # Prepare calibration data for both original and cleaned data
    X_orig = clean_matches[value_col].values.reshape(-1, 1)
    if is_log_scale:
        y_orig = np.log10(clean_matches['insitu_chl'].values)
    else:
        y_orig = clean_matches['insitu_chl'].values
    
    # Remove any invalid values from original data
    valid_idx_orig = np.isfinite(X_orig.flatten()) & np.isfinite(y_orig)
    X_orig = X_orig[valid_idx_orig]
    y_orig = y_orig[valid_idx_orig]
    
    # Prepare cleaned calibration data
    X_clean = clean_matches_no_outliers[value_col].values.reshape(-1, 1)
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
    
    return {
        'sensor': sensor_name,
        'original': {
            'model': model_orig,
            'X': X_orig,
            'y': y_orig,
            'r2': r2_orig,
            'rmse': rmse_orig,
            'n_matches': len(X_orig)
        },
        'cleaned': {
            'model': model_clean,
            'X': X_clean,
            'y': y_clean,
            'r2': r2_clean,
            'rmse': rmse_clean,
            'n_matches': len(X_clean)
        },
        'outlier_results': outlier_results,
        'is_log_scale': is_log_scale
    }


def apply_calibration(data, value_col, calibration_result, output_col='chl_calibrated'):
    """Apply calibration to satellite data"""
    if calibration_result is None:
        return data
    
    result = data.copy()
    valid_data = result.dropna(subset=[value_col])
    
    model = calibration_result['cleaned']['model']
    is_log_scale = calibration_result['is_log_scale']
    
    if is_log_scale:
        # OCx approach: log10(Chl) = a * value + b
        log_chl = valid_data[value_col] * model.coef_[0] + model.intercept_
        result.loc[valid_data.index, output_col] = np.power(10, log_chl)
    else:
        # Direct approach: Chl = a * value + b
        result.loc[valid_data.index, output_col] = (
            valid_data[value_col] * model.coef_[0] + model.intercept_
        )
    
    # Filter reasonable chlorophyll values
    result = result[
        (result[output_col] > 0) & 
        (result[output_col] <= 500)
    ].copy()
    
    return result


# ==============================================================================
# PLOTTING FUNCTIONS (from Calibrate_MODIS_Multi_Method.py)
# ==============================================================================

# Define sensor color and marker styles
SENSOR_STYLES = {
    'MODIS-Terra': {'color': 'orange', 'marker': 'o'},
    'MODIS-Aqua': {'color': 'green', 'marker': 'o'},
    'MODIS-Aqua-250m': {'color': 'blue', 'marker': '^'},
    'Terra': {'color': 'orange', 'marker': 'o'},
    'Aqua': {'color': 'green', 'marker': 'o'}
}


def create_outlier_comparison_plot(calibration_result, sensor_name, method_name=''):
    """Create comprehensive plots showing before/after outlier removal"""
    
    X_orig = calibration_result['original']['X']
    y_orig = calibration_result['original']['y']
    X_clean = calibration_result['cleaned']['X']
    y_clean = calibration_result['cleaned']['y']
    model_orig = calibration_result['original']['model']
    model_clean = calibration_result['cleaned']['model']
    r2_orig = calibration_result['original']['r2']
    rmse_orig = calibration_result['original']['rmse']
    r2_clean = calibration_result['cleaned']['r2']
    rmse_clean = calibration_result['cleaned']['rmse']
    outlier_results = calibration_result['outlier_results']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sensor_name} {method_name} Calibration: Outlier Analysis', fontsize=16, fontweight='bold')
    
    # Define consistent axis ranges
    x_min = min(X_orig.min(), X_clean.min()) - 0.1
    x_max = max(X_orig.max(), X_clean.max()) + 0.1
    
    # Row 1: Original data
    # Original scatter plot
    axes[0, 0].scatter(X_orig.flatten(), np.power(10, y_orig), alpha=0.6, s=30, color='blue', label='Data points')
    if outlier_results['n_outliers_removed'] > 0:
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
    filename = f'{sensor_name}_{method_clean}_outlier_analysis.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Outlier analysis plot saved: {filename}")


def create_time_series_plot(data_dict, insitu_data, title, filename, lake_name, ylim=(0, 350)):
    """
    Create comprehensive time series plot with satellite and in situ data
    
    Args:
        data_dict: Dictionary with sensor names as keys and dataframes as values
        insitu_data: DataFrame with in situ chlorophyll measurements
        title: Plot title
        filename: Output filename
        lake_name: Name of the lake (for labeling)
        ylim: Tuple of (min, max) for y-axis limits. Default (0, 350)
    """
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Plot in situ data
    if insitu_data is not None and len(insitu_data) > 0:
        # Determine the date column
        date_col = 'datetime' if 'datetime' in insitu_data.columns else 'date'
        chl_col = 'chlorophyll_a' if 'chlorophyll_a' in insitu_data.columns else 'chlorophyll_ugL'
        
        if chl_col in insitu_data.columns:
            label = 'In situ data'
            # Plot red square markers
            ax.scatter(insitu_data[date_col], insitu_data[chl_col], 
                      c='red', marker='s', s=12, alpha=0.7, label=label, zorder=1)
    
    # Plot satellite data
    for sensor_name, sensor_data in data_dict.items():
        if sensor_data is not None and len(sensor_data) > 0:
            # Get the appropriate style
            style = SENSOR_STYLES.get(sensor_name, {'color': 'gray', 'marker': 'o'})
            
            # Determine the columns to use
            date_col = 'datetime' if 'datetime' in sensor_data.columns else 'date'
            value_col = 'chl_calibrated' if 'chl_calibrated' in sensor_data.columns else 'chlorophyll_a'
            
            if value_col in sensor_data.columns:
                alpha_val = 0.6
                ax.scatter(sensor_data[date_col], sensor_data[value_col],
                          c=style['color'], marker=style['marker'], 
                          s=20, alpha=alpha_val, 
                          label=sensor_name, 
                          zorder=3)
    
    # Formatting
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Chlorophyll-a (µg/L)', fontsize=14)
    ax.set_xlabel('Date', fontsize=14)
    
    # Set y-axis limits
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    else:
        # Auto-scale based on data if no limits specified
        all_values = []
        if insitu_data is not None and len(insitu_data) > 0:
            chl_col = 'chlorophyll_a' if 'chlorophyll_a' in insitu_data.columns else 'chlorophyll_ugL'
            if chl_col in insitu_data.columns:
                all_values.extend(insitu_data[chl_col].dropna().values)
        
        for sensor_data in data_dict.values():
            if sensor_data is not None and len(sensor_data) > 0:
                value_col = 'chl_calibrated' if 'chl_calibrated' in sensor_data.columns else 'chlorophyll_a'
                if value_col in sensor_data.columns:
                    all_values.extend(sensor_data[value_col].dropna().values)
        
        if all_values:
            max_val = np.percentile(all_values, 95)
            ax.set_ylim(0, max_val * 1.1)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    
    # Improve date formatting
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved: {filename}")


def combine_modis_sensors(terra_data, aqua_data, date_col='datetime', value_col='chl_calibrated'):
    """Combine MODIS Terra and Aqua data into a single dataset"""
    if terra_data is None and aqua_data is None:
        return None
    
    # Prepare dataframes with sensor labels
    combined_data = []
    
    if terra_data is not None and len(terra_data) > 0:
        terra_subset = terra_data[[date_col, value_col]].copy()
        terra_subset['sensor'] = 'MODIS-Terra'
        combined_data.append(terra_subset)
    
    if aqua_data is not None and len(aqua_data) > 0:
        aqua_subset = aqua_data[[date_col, value_col]].copy()
        aqua_subset['sensor'] = 'MODIS-Aqua'
        combined_data.append(aqua_subset)
    
    if not combined_data:
        return None
    
    # Combine and sort
    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df = combined_df.sort_values(date_col).reset_index(drop=True)
    
    return combined_df


# ==============================================================================
# DATA LOADING FUNCTIONS
# ==============================================================================

def load_detroit_ysi_data():
    """Load Detroit Lake YSI in situ chlorophyll data"""
    try:
        excel_path = '../Data/Detroit/In Situ/CityofSalem_YSI_RawData.xlsx'
        
        # Read the Excel file
        df = pd.read_excel(excel_path, sheet_name='Sheet1')
        
        # Extract relevant columns
        detroit_insitu = df[['DateTime', 'Chl ug/L', 'Site ID (new)', 'Lat', 'Lon']].copy()
        detroit_insitu.columns = ['datetime', 'chlorophyll_ugL', 'site_id', 'latitude', 'longitude']
        
        # Clean the data
        detroit_insitu = detroit_insitu.dropna(subset=['datetime', 'chlorophyll_ugL'])
        detroit_insitu = detroit_insitu[
            (detroit_insitu['chlorophyll_ugL'] > 0) & 
            (detroit_insitu['chlorophyll_ugL'] <= 500)
        ].copy()
        
        # Convert date
        detroit_insitu['datetime'] = pd.to_datetime(detroit_insitu['datetime'], errors='coerce')
        detroit_insitu = detroit_insitu.dropna(subset=['datetime'])
        detroit_insitu = detroit_insitu.sort_values('datetime').reset_index(drop=True)
        
        print(f"Detroit YSI data loaded: {len(detroit_insitu)} measurements")
        print(f"Date range: {detroit_insitu['datetime'].min()} to {detroit_insitu['datetime'].max()}")
        print(f"Chlorophyll range: {detroit_insitu['chlorophyll_ugL'].min():.1f} - {detroit_insitu['chlorophyll_ugL'].max():.1f} µg/L")
        
        return detroit_insitu[['datetime', 'chlorophyll_ugL']].copy()
        
    except Exception as e:
        print(f"Warning: Could not load Detroit YSI data ({e})")
        return None


# ==============================================================================
# MATCHING FUNCTIONS
# ==============================================================================

def match_satellite_insitu(satellite_df, insitu_df, sat_value_col, temporal_window_hours=3):
    """
    Match satellite and in situ observations within specified time tolerance.
    """
    matches = []
    temporal_window = timedelta(hours=temporal_window_hours)
    
    # Determine the date column in both dataframes
    sat_date_col = 'datetime' if 'datetime' in satellite_df.columns else 'date'
    insitu_date_col = 'datetime' if 'datetime' in insitu_df.columns else 'date'
    insitu_chl_col = 'chlorophyll_a' if 'chlorophyll_a' in insitu_df.columns else 'chlorophyll_ugL'
    
    for _, insitu_row in insitu_df.iterrows():
        insitu_time = pd.to_datetime(insitu_row[insitu_date_col])
        
        # Find satellite observations within temporal window
        sat_times = pd.to_datetime(satellite_df[sat_date_col])
        time_diff = abs(sat_times - insitu_time)
        close_sat = satellite_df[time_diff <= temporal_window]
        
        if len(close_sat) > 0:
            # Use closest satellite observation
            closest_idx = time_diff[close_sat.index].argmin()
            sat_row = close_sat.iloc[closest_idx]
            
            matches.append({
                'date': insitu_time,
                'datetime': insitu_time,
                'satellite_value': sat_row[sat_value_col],
                'insitu_chl': insitu_row[insitu_chl_col],
                'time_diff_hours': time_diff[close_sat.index].iloc[closest_idx].total_seconds() / 3600,
                'sensor': sat_row.get('sensor', 'Unknown')
            })
    
    return pd.DataFrame(matches)


# ==============================================================================
# EXPORT FUNCTIONS
# ==============================================================================

def export_calibrated_data(data_dict, lake_name, method_name, output_dir='./'):
    """Export calibrated MODIS data to CSV files"""
    
    import os
    print(f"\nExporting {lake_name} {method_name} calibrated data to CSV files...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create method-specific suffix
    method_suffix = method_name.replace(' ', '_').replace('/', '_')
    
    # Export individual sensor files
    exported_files = []
    for sensor_name, data in data_dict.items():
        if data is not None and len(data) > 0:
            # Prepare data for export
            export_cols = ['datetime', 'chl_calibrated']
            if 'date' in data.columns:
                export_cols.append('date')
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
            exported_files.append(filepath)
    
    return exported_files