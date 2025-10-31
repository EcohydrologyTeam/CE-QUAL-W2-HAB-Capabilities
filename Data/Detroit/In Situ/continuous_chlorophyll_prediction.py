"""
Continuous daily chlorophyll predictions with gap-filling
Extends the discrete measurements to continuous time series
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def create_continuous_timeseries():
    """Create continuous daily time series with interpolated features"""
    
    print("Creating continuous daily time series...")
    
    # Load the discrete results
    results = pd.read_csv('chlorophyll_qc_results.csv')
    results['Date'] = pd.to_datetime(results['Date'])
    
    # Get date range for each site
    continuous_dfs = []
    
    for site in results['Site ID (new)'].unique():
        site_data = results[results['Site ID (new)'] == site].copy()
        
        # Create continuous date range
        date_range = pd.date_range(
            start=site_data['Date'].min(),
            end=site_data['Date'].max(),
            freq='D'
        )
        
        # Create continuous dataframe
        continuous = pd.DataFrame({
            'Date': date_range,
            'Site ID (new)': site
        })
        
        # Merge with existing measurements
        continuous = continuous.merge(site_data, on=['Date', 'Site ID (new)'], how='left')
        
        # Interpolate environmental variables
        interp_vars = ['DO mg/L_mean', 'DO %_mean', 'Temp °C_mean', 'pH_mean', 
                      'DO_diel', 'DO_sat_anomaly']
        
        for var in interp_vars:
            if var in continuous.columns:
                # Linear interpolation with limit of 7 days
                continuous[f'{var}_interp'] = continuous[var].interpolate(
                    method='linear', limit=7
                )
                # Forward/backward fill for longer gaps
                continuous[f'{var}_interp'] = continuous[f'{var}_interp'].ffill(limit=14).bfill(limit=14)
                
                # Track interpolation quality
                continuous[f'{var}_is_measured'] = continuous[var].notna()
        
        # Add seasonality (always available)
        continuous['day_of_year'] = continuous['Date'].dt.dayofyear
        continuous['sin_doy'] = np.sin(2 * np.pi * continuous['day_of_year'] / 365)
        continuous['cos_doy'] = np.cos(2 * np.pi * continuous['day_of_year'] / 365)
        
        # Calculate lagged features on interpolated data
        for lag in [1, 3, 7]:
            continuous[f'DO_diel_interp_lag{lag}'] = continuous['DO_diel_interp'].shift(lag)
            continuous[f'DO_sat_anomaly_interp_lag{lag}'] = continuous['DO_sat_anomaly_interp'].shift(lag)
        
        continuous_dfs.append(continuous)
    
    # Combine all sites
    full_continuous = pd.concat(continuous_dfs, ignore_index=True)
    
    return full_continuous

def apply_model_to_continuous(continuous_df):
    """Apply the trained model to continuous data"""
    
    print("Applying model to continuous time series...")
    
    # First, retrain model on original discrete data
    from simple_chlorophyll_qc import prepare_data, compute_daily_features, train_proxy_model
    
    ysi, algae, nutrients = prepare_data()
    daily = compute_daily_features(ysi, algae, nutrients)
    model, scaler, features = train_proxy_model(daily)
    
    # Map interpolated features to model features
    feature_mapping = {
        'DO_diel': 'DO_diel_interp',
        'DO_sat_anomaly': 'DO_sat_anomaly_interp',
        'DO mg/L_mean': 'DO mg/L_mean_interp',
        'Temp °C_mean': 'Temp °C_mean_interp',
        'pH_mean': 'pH_mean_interp',
        'DO_diel_lag1': 'DO_diel_interp_lag1',
        'DO_diel_lag3': 'DO_diel_interp_lag3',
        'DO_sat_anomaly_lag1': 'DO_sat_anomaly_interp_lag1',
        'sin_doy': 'sin_doy',
        'cos_doy': 'cos_doy'
    }
    
    # Prepare features for continuous prediction
    X_continuous = pd.DataFrame()
    for orig_feat in features:
        if orig_feat in feature_mapping:
            mapped_feat = feature_mapping[orig_feat]
            if mapped_feat in continuous_df.columns:
                X_continuous[orig_feat] = continuous_df[mapped_feat]
            else:
                X_continuous[orig_feat] = 0  # Default if not available
        else:
            X_continuous[orig_feat] = 0  # Default for missing features
    
    # Handle missing values
    X_continuous = X_continuous.fillna(0)
    X_continuous = X_continuous.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale and predict
    X_scaled = scaler.transform(X_continuous)
    log_pred = model.predict(X_scaled)
    continuous_df['Chl_continuous'] = 10**log_pred - 1
    
    # Calculate confidence based on data quality
    # Higher confidence when more variables are measured (not interpolated)
    measured_vars = [col for col in continuous_df.columns if '_is_measured' in col]
    if measured_vars:
        continuous_df['data_quality'] = continuous_df[measured_vars].mean(axis=1)
    else:
        continuous_df['data_quality'] = 0.5
    
    # Adjust predictions based on data quality
    continuous_df['Chl_continuous_lower'] = continuous_df['Chl_continuous'] * (0.5 + 0.5 * continuous_df['data_quality'])
    continuous_df['Chl_continuous_upper'] = continuous_df['Chl_continuous'] * (1.0 + 0.5 * (1 - continuous_df['data_quality']))
    
    return continuous_df

def plot_continuous_predictions(continuous_df, site='BCR'):
    """Plot continuous predictions with uncertainty"""
    
    site_data = continuous_df[continuous_df['Site ID (new)'] == site].copy()
    site_data = site_data.sort_values('Date')
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Continuous chlorophyll predictions
    ax1 = axes[0]
    
    # Continuous prediction with confidence bands
    ax1.fill_between(site_data['Date'], 
                     site_data['Chl_continuous_lower'],
                     site_data['Chl_continuous_upper'],
                     alpha=0.3, color='green', label='Prediction uncertainty')
    ax1.plot(site_data['Date'], site_data['Chl_continuous'], 
             'g-', label='Continuous prediction', linewidth=1)
    
    # Overlay measured points
    measured = site_data[site_data['Chl ug/L_mean'].notna()]
    ax1.scatter(measured['Date'], measured['Chl ug/L_mean'], 
               color='blue', s=20, label='YSI measurements', zorder=5)
    
    # Overlay algae counts
    anchors = site_data[site_data['Chl_from_algae'].notna()]
    if len(anchors) > 0:
        ax1.scatter(anchors['Date'], anchors['Chl_from_algae'], 
                   color='red', s=30, marker='^', label='Algae counts', zorder=5)
    
    ax1.set_ylabel('Chlorophyll-a (µg/L)')
    ax1.set_title(f'Continuous Chlorophyll Predictions - Site {site}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, max(100, site_data['Chl_continuous_upper'].quantile(0.95))])
    
    # Plot 2: Data quality indicator
    ax2 = axes[1]
    ax2.fill_between(site_data['Date'], 0, site_data['data_quality'], 
                     alpha=0.5, color='orange')
    ax2.set_ylabel('Data Quality\n(1=measured, 0=interpolated)')
    ax2.set_ylim([0, 1])
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Interpolation Quality Score')
    
    # Plot 3: Environmental drivers
    ax3 = axes[2]
    ax3.plot(site_data['Date'], site_data['DO_diel_interp'], 
             'b-', label='DO diel amplitude', alpha=0.7)
    
    ax3_twin = ax3.twinx()
    ax3_twin.plot(site_data['Date'], site_data['Temp °C_mean_interp'], 
                  'r-', label='Temperature', alpha=0.7)
    
    ax3.set_ylabel('DO Amplitude (mg/L)', color='blue')
    ax3_twin.set_ylabel('Temperature (°C)', color='red')
    ax3.set_xlabel('Date')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3_twin.tick_params(axis='y', labelcolor='red')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Environmental Drivers (interpolated)')
    
    # Add legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    """Generate continuous predictions"""
    
    print("\n=== CONTINUOUS CHLOROPHYLL PREDICTION ===\n")
    
    # Create continuous time series
    continuous = create_continuous_timeseries()
    
    # Apply model
    continuous = apply_model_to_continuous(continuous)
    
    # Save results
    continuous.to_csv('chlorophyll_continuous.csv', index=False)
    print(f"\nSaved continuous predictions to 'chlorophyll_continuous.csv'")
    
    # Summary statistics
    print("\n=== SUMMARY ===")
    for site in continuous['Site ID (new)'].unique():
        site_data = continuous[continuous['Site ID (new)'] == site]
        total_days = len(site_data)
        measured_days = site_data['Chl ug/L_mean'].notna().sum()
        print(f"{site}: {total_days} continuous days ({measured_days} measured, {total_days-measured_days} interpolated)")
    
    # Create plots
    print("\nGenerating visualizations...")
    for site in ['BCR', 'LB']:
        if site in continuous['Site ID (new)'].unique():
            fig = plot_continuous_predictions(continuous, site)
            filename = f'continuous_chlorophyll_{site}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
    
    return continuous

if __name__ == "__main__":
    continuous_results = main()
    plt.show()