"""
MODIS Multi-Term Calibration Utilities
=======================================
Extended version with multi-term regression model:
log10(Chl_insitu) = α0 + α1*log10(B4/B1) + α2*(B2/B1)

Key improvements:
- 5-day matching window for satellite-insitu pairing
- Multi-term regression with Green/Red and NIR/Red terms
- Enhanced statistical analysis
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


# Import all functions from original utilities
from modis_calibration_utils import (
    detect_outliers_iqr,
    detect_outliers_zscore,
    detect_outliers_cooks_distance,
    detect_outliers_combined,
    create_outlier_comparison_plot,
    create_time_series_plot,
    combine_modis_sensors,
    export_calibrated_data,
    load_detroit_ysi_data,
    SENSOR_STYLES
)


# ==============================================================================
# ENHANCED MATCHING FUNCTION WITH 5-DAY WINDOW
# ==============================================================================

def match_satellite_insitu_5day(satellite_df, insitu_df, temporal_window_days=5):
    """
    Match satellite and in situ observations within specified time tolerance.
    For each MODIS reading, find the closest in situ reading within the window.
    
    Args:
        satellite_df: DataFrame with satellite observations
        insitu_df: DataFrame with in situ observations  
        temporal_window_days: Maximum days between observations (default 5)
    
    Returns:
        DataFrame with matched observations
    """
    matches = []
    temporal_window = timedelta(days=temporal_window_days)
    
    # Determine the date columns
    sat_date_col = 'datetime' if 'datetime' in satellite_df.columns else 'date'
    insitu_date_col = 'datetime' if 'datetime' in insitu_df.columns else 'date'
    insitu_chl_col = 'chlorophyll_a' if 'chlorophyll_a' in insitu_df.columns else 'chlorophyll_ugL'
    
    # Ensure datetime format
    satellite_df[sat_date_col] = pd.to_datetime(satellite_df[sat_date_col])
    insitu_df[insitu_date_col] = pd.to_datetime(insitu_df[insitu_date_col])
    
    for _, sat_row in satellite_df.iterrows():
        sat_time = sat_row[sat_date_col]
        
        # Find in situ observations within temporal window
        time_diff = abs(insitu_df[insitu_date_col] - sat_time)
        within_window = time_diff <= temporal_window
        
        if within_window.any():
            # Get the closest in situ observation
            closest_idx = time_diff[within_window].idxmin()
            insitu_row = insitu_df.loc[closest_idx]
            
            matches.append({
                'datetime': sat_time,
                'insitu_datetime': insitu_row[insitu_date_col],
                'band1_red': sat_row.get('band1_red', np.nan),
                'band2_nir': sat_row.get('band2_nir', np.nan),
                'band4_green': sat_row.get('band4_green', np.nan),
                'green_red_ratio': sat_row.get('band4_green', np.nan) / sat_row.get('band1_red', 1),
                'nir_red_ratio': sat_row.get('band2_nir', np.nan) / sat_row.get('band1_red', 1),
                'log_green_red': np.log10(sat_row.get('band4_green', np.nan) / sat_row.get('band1_red', 1)),
                'insitu_chl': insitu_row[insitu_chl_col],
                'time_diff_days': time_diff[closest_idx].days,
                'sensor': sat_row.get('sensor', 'Unknown')
            })
    
    matched_df = pd.DataFrame(matches)
    
    # Remove invalid values
    if len(matched_df) > 0:
        matched_df = matched_df[
            (matched_df['insitu_chl'] > 0) & 
            np.isfinite(matched_df['band1_red']) &
            np.isfinite(matched_df['band2_nir']) &
            np.isfinite(matched_df['band4_green']) &
            (matched_df['band1_red'] > 0)
        ]
    
    return matched_df


# ==============================================================================
# MULTI-TERM CALIBRATION FUNCTION
# ==============================================================================

def calibrate_multiterm_model(matched_data, sensor_name="MODIS", remove_outliers=True):
    """
    Calibrate multi-term model: log10(Chl) = α0 + α1*log10(B4/B1) + α2*(B2/B1)
    
    Args:
        matched_data: DataFrame with matched satellite-insitu observations
        sensor_name: Name for reporting
        remove_outliers: Whether to remove outliers
    
    Returns:
        Dictionary with calibration results
    """
    print(f"\n{'='*60}")
    print(f"MULTI-TERM CALIBRATION: {sensor_name}")
    print(f"{'='*60}")
    
    # Prepare features and target
    X = pd.DataFrame()
    X['log_green_red'] = np.log10(matched_data['band4_green'] / matched_data['band1_red'])
    X['nir_red'] = matched_data['band2_nir'] / matched_data['band1_red']
    
    y = np.log10(matched_data['insitu_chl'])
    
    # Remove invalid values
    valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X_clean = X[valid_mask].values
    y_clean = y[valid_mask].values
    matched_clean = matched_data[valid_mask].copy()
    
    print(f"Initial clean matches: {len(matched_clean)}")
    
    # Outlier detection if requested
    outlier_results = {'n_outliers_removed': 0}
    
    if remove_outliers and len(matched_clean) >= 20:
        # Create temporary dataframe for outlier detection
        temp_df = matched_clean.copy()
        temp_df['satellite_value'] = X_clean[:, 0]  # Use log_green_red as primary
        
        # Detect outliers
        _, outlier_flags, outlier_info = detect_outliers_combined(temp_df)
        
        # Remove outliers
        outliers_to_remove = outlier_flags['multiple_methods']
        n_outliers = outliers_to_remove.sum()
        
        print(f"Outliers detected: {n_outliers}")
        
        if n_outliers > 0:
            clean_mask = ~outliers_to_remove
            X_no_outliers = X_clean[clean_mask]
            y_no_outliers = y_clean[clean_mask]
            matched_no_outliers = matched_clean[clean_mask].copy()
            
            outlier_results = {
                'n_outliers_removed': n_outliers,
                'outliers_removed': matched_clean[outliers_to_remove],
                'outlier_flags': outlier_flags,
                'outlier_info': outlier_info
            }
        else:
            X_no_outliers = X_clean
            y_no_outliers = y_clean
            matched_no_outliers = matched_clean
    else:
        X_no_outliers = X_clean
        y_no_outliers = y_clean
        matched_no_outliers = matched_clean
        print("Skipping outlier detection")
    
    # Fit models (original and cleaned)
    model_orig = HuberRegressor()
    model_orig.fit(X_clean, y_clean)
    
    model_clean = HuberRegressor()
    model_clean.fit(X_no_outliers, y_no_outliers)
    
    # Calculate metrics
    y_pred_orig = model_orig.predict(X_clean)
    r2_orig = r2_score(y_clean, y_pred_orig)
    rmse_orig = np.sqrt(mean_squared_error(y_clean, y_pred_orig))
    
    y_pred_clean = model_clean.predict(X_no_outliers)
    r2_clean = r2_score(y_no_outliers, y_pred_clean)
    rmse_clean = np.sqrt(mean_squared_error(y_no_outliers, y_pred_clean))
    
    # Extract coefficients
    alpha0 = model_clean.intercept_
    alpha1 = model_clean.coef_[0]  # log10(green/red) coefficient
    alpha2 = model_clean.coef_[1]  # nir/red coefficient
    
    print(f"\nCalibration Results:")
    print(f"  Equation: log10(Chl) = {alpha0:.3f} + {alpha1:.3f}*log10(B4/B1) + {alpha2:.3f}*(B2/B1)")
    print(f"  Original data: {len(X_clean)} matches, R² = {r2_orig:.3f}, RMSE = {rmse_orig:.3f}")
    print(f"  Cleaned data: {len(X_no_outliers)} matches, R² = {r2_clean:.3f}, RMSE = {rmse_clean:.3f}")
    print(f"  Improvement: ΔR² = {r2_clean - r2_orig:+.3f}")
    
    return {
        'sensor': sensor_name,
        'model_type': 'multiterm',
        'equation': f'log10(Chl) = {alpha0:.3f} + {alpha1:.3f}*log10(B4/B1) + {alpha2:.3f}*(B2/B1)',
        'original': {
            'model': model_orig,
            'X': X_clean,
            'y': y_clean,
            'r2': r2_orig,
            'rmse': rmse_orig,
            'n_matches': len(X_clean)
        },
        'cleaned': {
            'model': model_clean,
            'X': X_no_outliers,
            'y': y_no_outliers,
            'r2': r2_clean,
            'rmse': rmse_clean,
            'n_matches': len(X_no_outliers),
            'alpha0': alpha0,
            'alpha1': alpha1,
            'alpha2': alpha2
        },
        'outlier_results': outlier_results,
        'matched_data': matched_no_outliers
    }


# ==============================================================================
# APPLY MULTI-TERM CALIBRATION
# ==============================================================================

def apply_multiterm_calibration(data, calibration_result, output_col='chl_multiterm'):
    """
    Apply multi-term calibration to satellite data
    
    Args:
        data: DataFrame with satellite observations
        calibration_result: Dictionary from calibrate_multiterm_model
        output_col: Name for output column
    
    Returns:
        DataFrame with calibrated chlorophyll values
    """
    if calibration_result is None:
        return data
    
    result = data.copy()
    
    # Get coefficients
    alpha0 = calibration_result['cleaned']['alpha0']
    alpha1 = calibration_result['cleaned']['alpha1']
    alpha2 = calibration_result['cleaned']['alpha2']
    
    # Calculate features
    log_green_red = np.log10(result['band4_green'] / result['band1_red'])
    nir_red = result['band2_nir'] / result['band1_red']
    
    # Apply equation: log10(Chl) = α0 + α1*log10(B4/B1) + α2*(B2/B1)
    log_chl = alpha0 + alpha1 * log_green_red + alpha2 * nir_red
    
    # Convert back from log scale
    result[output_col] = np.power(10, log_chl)
    
    # Filter reasonable values
    result = result[
        (result[output_col] > 0) & 
        (result[output_col] <= 500) &
        np.isfinite(result[output_col])
    ].copy()
    
    return result


# ==============================================================================
# ENHANCED PLOTTING FOR MULTI-TERM MODEL
# ==============================================================================

def plot_multiterm_calibration(calibration_result, sensor_name="MODIS"):
    """
    Create comprehensive plots for multi-term calibration results
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{sensor_name} Multi-Term Calibration Analysis', fontsize=16, fontweight='bold')
    
    # Get data
    X_orig = calibration_result['original']['X']
    y_orig = calibration_result['original']['y']
    X_clean = calibration_result['cleaned']['X']
    y_clean = calibration_result['cleaned']['y']
    
    model_orig = calibration_result['original']['model']
    model_clean = calibration_result['cleaned']['model']
    
    # Plot 1: Original data scatter
    ax = axes[0, 0]
    y_pred_orig = model_orig.predict(X_orig)
    scatter = ax.scatter(10**y_pred_orig, 10**y_orig, alpha=0.6, s=30, c='blue')
    ax.plot([0.1, 1000], [0.1, 1000], 'r--', label='1:1 line')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted Chlorophyll (µg/L)')
    ax.set_ylabel('Observed Chlorophyll (µg/L)')
    ax.set_title(f'Original Data (R² = {calibration_result["original"]["r2"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Residuals original
    ax = axes[0, 1]
    residuals_orig = y_orig - y_pred_orig
    ax.scatter(y_pred_orig, residuals_orig, alpha=0.6, s=30, c='blue')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted log10(Chl)')
    ax.set_ylabel('Residuals')
    ax.set_title('Original Data Residuals')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Feature importance
    ax = axes[0, 2]
    feature_names = ['log10(Green/Red)', 'NIR/Red']
    coefficients = model_clean.coef_
    colors = ['green', 'red']
    bars = ax.bar(feature_names, coefficients, color=colors, alpha=0.7)
    ax.set_ylabel('Coefficient Value')
    ax.set_title('Feature Coefficients')
    ax.grid(True, alpha=0.3)
    # Add value labels on bars
    for bar, coef in zip(bars, coefficients):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{coef:.3f}', ha='center', va='bottom')
    
    # Plot 4: Cleaned data scatter
    ax = axes[1, 0]
    y_pred_clean = model_clean.predict(X_clean)
    scatter = ax.scatter(10**y_pred_clean, 10**y_clean, alpha=0.6, s=30, c='green')
    ax.plot([0.1, 1000], [0.1, 1000], 'r--', label='1:1 line')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Predicted Chlorophyll (µg/L)')
    ax.set_ylabel('Observed Chlorophyll (µg/L)')
    ax.set_title(f'Cleaned Data (R² = {calibration_result["cleaned"]["r2"]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Residuals cleaned
    ax = axes[1, 1]
    residuals_clean = y_clean - y_pred_clean
    ax.scatter(y_pred_clean, residuals_clean, alpha=0.6, s=30, c='green')
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted log10(Chl)')
    ax.set_ylabel('Residuals')
    ax.set_title('Cleaned Data Residuals')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Q-Q plot
    ax = axes[1, 2]
    stats.probplot(residuals_clean, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Cleaned Residuals)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{sensor_name}_multiterm_calibration_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Multi-term calibration plot saved: {sensor_name}_multiterm_calibration_analysis.png")