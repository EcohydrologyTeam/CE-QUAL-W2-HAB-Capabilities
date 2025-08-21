"""
Test MODIS calibration using UKL in situ data with outlier detection and removal
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
from scipy.stats import zscore
from load_ukl_data import load_ukl_insitu_data, filter_ukl_data_for_satellite_calibration

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

def read_data(inpath):
    df = pd.read_csv(inpath)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.sort_values('date', inplace=True)
    return df

def match_satellite_insitu(satellite_df, insitu_df, sat_value_col, tolerance_days=5):
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

def calibrate_modis_to_insitu(modis_df, insitu_df, sensor_name, remove_outliers=True, outlier_methods=['iqr', 'zscore', 'cooks']):
    """Test MODIS calibration with UKL data, including outlier detection and removal"""
    print(f"Testing MODIS {sensor_name} calibration with UKL data...")
    
    # Match MODIS data with in situ
    matched = match_satellite_insitu(modis_df, insitu_df, 'log_ratio', tolerance_days=5)
    
    print(f"Initial matches: {len(matched)}")
    
    if len(matched) < 5:
        print(f"ERROR: Only {len(matched)} matches found - not enough for calibration")
        return None
    
    # Remove invalid values
    clean_matches = matched[matched['insitu_chl'] > 0].copy()
    
    print(f"Clean matches: {len(clean_matches)} (removed {len(matched) - len(clean_matches)} negative Chl values)")
    if len(clean_matches) > 0:
        print(f"log_ratio range: {clean_matches['satellite_value'].min():.3f} to {clean_matches['satellite_value'].max():.3f}")
        print(f"In situ Chl range: {clean_matches['insitu_chl'].min():.1f} to {clean_matches['insitu_chl'].max():.1f} µg/L")
    
    if len(clean_matches) < 5:
        print("ERROR: Not enough clean matches for calibration")
        return None
    
    # Outlier detection
    outlier_results = {}
    if remove_outliers and len(clean_matches) >= 10:  # Need enough data for meaningful outlier detection
        print(f"\n--- Outlier Detection ---")
        matched_clean, outlier_flags, outlier_info = detect_outliers_combined(clean_matches, methods=outlier_methods)
        
        for method in outlier_methods:
            if method in outlier_info:
                print(f"{method.upper()} method: {outlier_info[method]['total_outliers']} outliers detected")
        
        # Use conservative approach: only remove if multiple methods agree
        outliers_to_remove = outlier_flags['multiple_methods']
        n_outliers = outliers_to_remove.sum()
        
        print(f"Total outliers to remove (multiple methods agree): {n_outliers}")
        
        if n_outliers > 0:
            clean_matches_no_outliers = matched_clean[~outliers_to_remove].copy()
            outlier_data = matched_clean[outliers_to_remove].copy()
            print(f"Remaining data points after outlier removal: {len(clean_matches_no_outliers)}")
            
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
        print("Skipping outlier detection (not enough data or disabled)")
        outlier_results['n_outliers_removed'] = 0
    
    # Prepare data for OCx calibration: log10(Chl) = a * log_ratio + b
    X_orig = clean_matches['satellite_value'].values.reshape(-1, 1)
    y_orig = np.log10(clean_matches['insitu_chl'].values)
    
    X_clean = clean_matches_no_outliers['satellite_value'].values.reshape(-1, 1)
    y_clean = np.log10(clean_matches_no_outliers['insitu_chl'].values)
    
    # Remove any remaining invalid values
    valid_idx_orig = np.isfinite(X_orig.flatten()) & np.isfinite(y_orig)
    X_orig = X_orig[valid_idx_orig]
    y_orig = y_orig[valid_idx_orig]
    
    valid_idx_clean = np.isfinite(X_clean.flatten()) & np.isfinite(y_clean)
    X_clean = X_clean[valid_idx_clean]
    y_clean = y_clean[valid_idx_clean]
    
    if len(X_clean) < 5:
        print(f"ERROR: Not enough valid data points after outlier removal")
        return None
    
    # Fit models - both with and without outliers for comparison
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
    
    print(f"\nMODIS {sensor_name} Calibration Results with REAL UKL Data:")
    print(f"Original data (with outliers):")
    print(f"  - Observations: {len(clean_matches)}")
    print(f"  - Slope: {model_orig.coef_[0]:.3f}")
    print(f"  - Intercept: {model_orig.intercept_:.3f}")
    print(f"  - R²: {r2_orig:.3f}")
    print(f"  - RMSE: {rmse_orig:.3f}")
    
    print(f"Cleaned data (outliers removed):")
    print(f"  - Observations: {len(clean_matches_no_outliers)} (removed {outlier_results['n_outliers_removed']} outliers)")
    print(f"  - Slope: {model_clean.coef_[0]:.3f}")
    print(f"  - Intercept: {model_clean.intercept_:.3f}")
    print(f"  - R²: {r2_clean:.3f}")
    print(f"  - RMSE: {rmse_clean:.3f}")
    print(f"  - Improvement in R²: {r2_clean - r2_orig:+.3f}")
    print(f"  - Improvement in RMSE: {rmse_orig - rmse_clean:+.3f}")
    
    # Create comprehensive comparison plot
    create_outlier_comparison_plot(X_orig, y_orig, X_clean, y_clean, model_orig, model_clean, 
                                 sensor_name, outlier_results, r2_orig, rmse_orig, r2_clean, rmse_clean)
    
    return {
        'sensor': f'MODIS_{sensor_name}',
        'original': {
            'n_matches': len(clean_matches),
            'slope': model_orig.coef_[0],
            'intercept': model_orig.intercept_,
            'r2': r2_orig,
            'rmse': rmse_orig,
            'model': model_orig,
            'matched_data': clean_matches
        },
        'cleaned': {
            'n_matches': len(clean_matches_no_outliers),
            'slope': model_clean.coef_[0],
            'intercept': model_clean.intercept_,
            'r2': r2_clean,
            'rmse': rmse_clean,
            'model': model_clean,
            'matched_data': clean_matches_no_outliers
        },
        'outlier_results': outlier_results
    }

def create_outlier_comparison_plot(X_orig, y_orig, X_clean, y_clean, model_orig, model_clean, 
                                 sensor_name, outlier_results, r2_orig, rmse_orig, r2_clean, rmse_clean):
    """Create comprehensive plots showing before/after outlier removal"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'MODIS {sensor_name} Calibration: Outlier Analysis', fontsize=16, fontweight='bold')
    
    # Define consistent axis ranges
    x_min = min(X_orig.min(), X_clean.min()) - 0.1
    x_max = max(X_orig.max(), X_clean.max()) + 0.1
    y_min_log = min(y_orig.min(), y_clean.min()) - 0.1
    y_max_log = max(y_orig.max(), y_clean.max()) + 0.1
    
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
    axes[0, 0].set_xlabel('MODIS log_ratio')
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
    axes[1, 0].set_xlabel('MODIS log_ratio')
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
    plt.savefig(f'MODIS_{sensor_name}_outlier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create summary statistics plot
    if outlier_results['n_outliers_removed'] > 0:
        create_outlier_summary_plot(outlier_results, sensor_name)

def create_outlier_summary_plot(outlier_results, sensor_name):
    """Create a summary plot showing outlier detection method comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'MODIS {sensor_name}: Outlier Detection Summary', fontsize=14, fontweight='bold')
    
    # Method comparison
    if 'outlier_info' in outlier_results:
        methods = []
        counts = []
        for method, info in outlier_results['outlier_info'].items():
            methods.append(method.upper())
            counts.append(info['total_outliers'])
        
        axes[0].bar(methods, counts, alpha=0.7, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0].set_ylabel('Number of Outliers Detected')
        axes[0].set_title('Outliers by Detection Method')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(counts):
            axes[0].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Cook's distance plot if available
    if 'outlier_info' in outlier_results and 'cooks' in outlier_results['outlier_info']:
        cooks_distances = outlier_results['outlier_info']['cooks']['distances']
        threshold = 4 / len(cooks_distances)
        
        axes[1].scatter(range(len(cooks_distances)), cooks_distances, alpha=0.6)
        axes[1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
        axes[1].set_xlabel('Observation Index')
        axes[1].set_ylabel("Cook's Distance")
        axes[1].set_title("Cook's Distance Plot")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'MODIS_{sensor_name}_outlier_summary.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("Testing MODIS calibration with UKL in situ data...")
    print("="*60)
    
    # Load UKL in situ data
    raw_data = load_ukl_insitu_data()
    if len(raw_data) == 0:
        print("ERROR: Could not load UKL data")
        return
    
    # Filter for satellite calibration
    insitu_data = filter_ukl_data_for_satellite_calibration(raw_data, min_chl=1.0, max_chl=500.0)
    
    print(f"\nIn situ data summary:")
    print(f"Records: {len(insitu_data)}")
    print(f"Date range: {insitu_data['date'].min()} to {insitu_data['date'].max()}")
    print(f"Chlorophyll range: {insitu_data['chlorophyll_ugL'].min():.1f} - {insitu_data['chlorophyll_ugL'].max():.1f} µg/L")
    
    # Load UKL MODIS data (V3 format with log_ratio)
    try:
        ukl_modis_terra = read_data('Klamath_MODIS_Terra_500m_ROI.csv')
        ukl_modis_aqua = read_data('Klamath_MODIS_Aqua_500m_ROI.csv')
        
        if 'log_ratio' not in ukl_modis_terra.columns:
            print("ERROR: V3 MODIS files with log_ratio not found")
            return
            
        # Add sensor labels
        ukl_modis_terra['sensor'] = 'Terra'
        ukl_modis_aqua['sensor'] = 'Aqua'
        
        print(f"\nMODIS Terra data: {len(ukl_modis_terra)} records")
        print(f"Date range: {ukl_modis_terra['date'].min()} to {ukl_modis_terra['date'].max()}")
        
        print(f"\nMODIS Aqua data: {len(ukl_modis_aqua)} records")
        print(f"Date range: {ukl_modis_aqua['date'].min()} to {ukl_modis_aqua['date'].max()}")
        
    except FileNotFoundError:
        print("ERROR: Could not load MODIS V3 data files")
        return
    
    # Check temporal overlap
    insitu_start = insitu_data['date'].min()
    insitu_end = insitu_data['date'].max()
    modis_start = max(ukl_modis_terra['date'].min(), ukl_modis_aqua['date'].min())
    modis_end = min(ukl_modis_terra['date'].max(), ukl_modis_aqua['date'].max())
    
    overlap_start = max(insitu_start, modis_start)
    overlap_end = min(insitu_end, modis_end)
    
    print(f"\nTemporal overlap analysis:")
    print(f"In situ: {insitu_start} to {insitu_end}")
    print(f"MODIS: {modis_start} to {modis_end}")
    print(f"Overlap: {overlap_start} to {overlap_end}")
    
    if overlap_start > overlap_end:
        print("ERROR: No temporal overlap between in situ and MODIS data!")
        return
    
    # Perform calibrations with data
    print(f"\n{'='*60}")
    print("CALIBRATION WITH REAL UKL DATA")
    print("="*60)
    
    # Filter data to overlap period
    insitu_overlap = insitu_data[
        (insitu_data['date'] >= overlap_start) & 
        (insitu_data['date'] <= overlap_end)
    ].copy()
    
    terra_overlap = ukl_modis_terra[
        (ukl_modis_terra['date'] >= overlap_start) & 
        (ukl_modis_terra['date'] <= overlap_end)
    ].copy()
    
    aqua_overlap = ukl_modis_aqua[
        (ukl_modis_aqua['date'] >= overlap_start) & 
        (ukl_modis_aqua['date'] <= overlap_end)
    ].copy()
    
    print(f"\nData in overlap period:")
    print(f"In situ: {len(insitu_overlap)} records")
    print(f"MODIS Terra: {len(terra_overlap)} records")
    print(f"MODIS Aqua: {len(aqua_overlap)} records")
    
    # Calibrate Terra
    terra_results = calibrate_modis_to_insitu(terra_overlap, insitu_overlap, 'Terra')
    
    # Calibrate Aqua
    aqua_results = calibrate_modis_to_insitu(aqua_overlap, insitu_overlap, 'Aqua')
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY OF REAL UKL DATA CALIBRATION")
    print("="*60)
    
    if terra_results:
        print(f"MODIS Terra:")
        print(f"  Original: R² = {terra_results['original']['r2']:.3f}, RMSE = {terra_results['original']['rmse']:.3f}")
        print(f"  Cleaned:  R² = {terra_results['cleaned']['r2']:.3f}, RMSE = {terra_results['cleaned']['rmse']:.3f}")
        print(f"  Outliers removed: {terra_results['outlier_results']['n_outliers_removed']}")
    else:
        print("MODIS Terra: FAILED")
        
    if aqua_results:
        print(f"MODIS Aqua:")
        print(f"  Original: R² = {aqua_results['original']['r2']:.3f}, RMSE = {aqua_results['original']['rmse']:.3f}")
        print(f"  Cleaned:  R² = {aqua_results['cleaned']['r2']:.3f}, RMSE = {aqua_results['cleaned']['rmse']:.3f}")
        print(f"  Outliers removed: {aqua_results['outlier_results']['n_outliers_removed']}")
    else:
        print("MODIS Aqua: FAILED")
    
    if terra_results or aqua_results:
        print(f"\nSUCCESS! Calibration completed with UKL data.")
    else:
        print("\nFAILURE: Could not complete any calibrations")

if __name__ == "__main__":
    main()