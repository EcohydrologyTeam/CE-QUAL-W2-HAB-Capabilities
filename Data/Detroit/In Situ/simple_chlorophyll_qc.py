"""
Simplified implementation following ChatGPT's recipe
Ready-to-run with minimal dependencies
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def prepare_data():
    """Step 1: Load and prepare data"""
    
    print("Loading data...")
    
    # Load YSI data
    ysi = pd.read_excel('CityofSalem_YSI_RawData.xlsx')
    ysi['Date'] = pd.to_datetime(ysi['DateTime']).dt.date
    ysi['Hour'] = pd.to_datetime(ysi['DateTime']).dt.hour
    
    # Load algae counts
    algae = pd.read_excel('CityofSalem_NutrientsAlgae_Raw.xlsx', sheet_name='AlgaeID_Enumeration')
    algae['Date'] = pd.to_datetime(algae['DATE']).dt.date
    
    # Load nutrients
    nutrients = pd.read_excel('CityofSalem_NutrientsAlgae_Raw.xlsx', sheet_name='Nutrients_WillowLakeLab')
    nutrients['Date'] = pd.to_datetime(nutrients['Date'], errors='coerce').dt.date
    
    return ysi, algae, nutrients

def compute_daily_features(ysi, algae, nutrients):
    """Step 2: Compute daily aggregates and features"""
    
    print("Computing daily features...")
    
    # === YSI Daily Aggregates ===
    daily = ysi.groupby(['Site ID (new)', 'Date']).agg({
        'Chl RFU': ['mean', 'std'],
        'Chl ug/L': 'mean',
        'DO mg/L': ['mean', 'min', 'max'],
        'DO %': 'mean',
        'Temp °C': 'mean',
        'pH': 'mean',
        'BGA-PC RFU': 'mean'
    }).reset_index()
    
    # Flatten columns
    daily.columns = ['_'.join(col).strip('_') for col in daily.columns]
    
    # DO diel amplitude (GPP proxy)
    daily['DO_diel'] = daily['DO mg/L_max'] - daily['DO mg/L_min']
    
    # DO saturation anomaly
    daily['DO_sat_anomaly'] = daily['DO %_mean'] - 100
    
    # === Night fluorescence (NPQ correction) ===
    night_ysi = ysi[(ysi['Hour'] >= 22) | (ysi['Hour'] <= 4)]
    night_fluor = night_ysi.groupby(['Site ID (new)', 'Date'])['Chl RFU'].mean().reset_index()
    night_fluor.columns = ['Site ID (new)', 'Date', 'Chl_RFU_night']
    daily = daily.merge(night_fluor, on=['Site ID (new)', 'Date'], how='left')
    
    # === Algae-based chlorophyll ===
    # Aggregate by site and date
    algae_daily = algae.groupby(['Sample Site', 'Date']).agg({
        'DENSITY (cells/mL) REP 1': 'sum',
        'TOTAL BV (um3/mL)': 'sum',
        'DIVISION': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    # Calculate division shares
    for div in ['Cyanobacteria', 'Bacillariophyta', 'Chlorophyta']:
        algae_daily[f'share_{div}'] = algae_daily['DIVISION'].apply(
            lambda x: x.get(div, 0) / sum(x.values()) if x else 0
        )
    
    # Simple Chl:biovolume conversion 
    # Typical range: 0.001-0.01 pg Chl/µm³ biovolume
    # Since TOTAL BV is in um3/mL, and we want µg/L:
    # 1 um3/mL = 1e-6 mm3/mL = 1e-9 g/L biovolume
    # With 0.005 pg Chl/um3 = 5e-15 g Chl/um3
    # So: um3/mL * 5e-15 * 1e9 = um3/mL * 5e-6 = µg Chl/L
    algae_daily['Chl_from_algae'] = algae_daily['TOTAL BV (um3/mL)'] * 5e-6
    
    # Log transforms (handle zeros and NaNs)
    algae_daily['log_cells'] = np.log10(algae_daily['DENSITY (cells/mL) REP 1'].fillna(1) + 1)
    algae_daily['log_biovolume'] = np.log10(algae_daily['TOTAL BV (um3/mL)'].fillna(1) + 1)
    
    # Merge with daily data
    daily = daily.merge(
        algae_daily[['Sample Site', 'Date', 'Chl_from_algae', 'log_cells', 
                     'log_biovolume', 'share_Cyanobacteria', 'share_Bacillariophyta']],
        left_on=['Site ID (new)', 'Date'],
        right_on=['Sample Site', 'Date'],
        how='left'
    )
    
    # === Nutrients ===
    nutrients_clean = nutrients[['Site Code', 'Date', 'NH3-ISE (mg/L), lo-level', 
                                 'NO3+NO2 (mg/L)', 'OP-Phos (mg/L)']].dropna(subset=['Date']).copy()
    
    # Clean nutrient values (handle detection limits and text)
    def clean_nutrient_value(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            # Remove < signs and treat as half detection limit
            x = x.strip().replace('< ', '').replace('<', '')
            try:
                return float(x) / 2  # Use half detection limit
            except:
                return np.nan
        return np.nan
    
    # Log transform
    for col in ['NH3-ISE (mg/L), lo-level', 'NO3+NO2 (mg/L)', 'OP-Phos (mg/L)']:
        nutrients_clean[col] = nutrients_clean[col].apply(clean_nutrient_value)
        nutrients_clean[f'log_{col}'] = np.log10(nutrients_clean[col].fillna(0.001) + 0.001)
    
    daily = daily.merge(
        nutrients_clean,
        left_on=['Site ID (new)', 'Date'],
        right_on=['Site Code', 'Date'],
        how='left'
    )
    
    # === Add lags (1-7 days) ===
    lag_features = ['DO_diel', 'DO_sat_anomaly', 'Temp °C_mean']
    
    for feat in lag_features:
        for lag in [1, 3, 7]:
            daily[f'{feat}_lag{lag}'] = daily.groupby('Site ID (new)')[feat].shift(lag)
    
    # === Seasonality ===
    daily['day_of_year'] = pd.to_datetime(daily['Date']).dt.dayofyear
    daily['sin_doy'] = np.sin(2 * np.pi * daily['day_of_year'] / 365)
    daily['cos_doy'] = np.cos(2 * np.pi * daily['day_of_year'] / 365)
    
    return daily

def train_proxy_model(daily_df):
    """Step 3: Train Elastic Net on anchor days"""
    
    print("Training proxy model...")
    
    # Filter to anchor days (where we have algae counts)
    anchor_df = daily_df[daily_df['Chl_from_algae'].notna()].copy()
    
    print(f"  Using {len(anchor_df)} anchor points")
    
    # Define features
    feature_cols = [
        # DO metrics
        'DO_diel', 'DO_sat_anomaly', 'DO mg/L_mean',
        # Environmental
        'Temp °C_mean', 'pH_mean',
        # Lagged features
        'DO_diel_lag1', 'DO_diel_lag3', 'DO_sat_anomaly_lag1',
        # Seasonality
        'sin_doy', 'cos_doy',
        # Fluorescence
        'Chl_RFU_night',
        # Algae composition (when available)
        'share_Cyanobacteria', 'share_Bacillariophyta'
    ]
    
    # Add nutrients if available
    nut_cols = ['log_NH3-ISE (mg/L), lo-level', 'log_NO3+NO2 (mg/L)', 'log_OP-Phos (mg/L)']
    feature_cols.extend([c for c in nut_cols if c in anchor_df.columns])
    
    # Select available features
    available_features = [f for f in feature_cols if f in anchor_df.columns]
    X = anchor_df[available_features].ffill().fillna(0)
    
    # Replace infinities with NaN then fill
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    y = np.log10(anchor_df['Chl_from_algae'] + 1)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Elastic Net with cross-validation
    model = ElasticNetCV(
        cv=TimeSeriesSplit(n_splits=3),
        alphas=np.logspace(-2, 1, 10),
        l1_ratio=[0.5, 0.7, 0.9],
        max_iter=2000
    )
    
    model.fit(X_scaled, y)
    
    # Calculate R²
    from sklearn.metrics import r2_score
    y_pred = model.predict(X_scaled)
    r2 = r2_score(y, y_pred)
    print(f"  Model R² on anchor points: {r2:.3f}")
    
    return model, scaler, available_features

def apply_model_and_fuse(daily_df, model, scaler, feature_cols):
    """Step 4: Apply model to all days and fuse with RFU"""
    
    print("Generating predictions for all days...")
    
    # Prepare features for all days
    X_all = daily_df[feature_cols].ffill().fillna(0)
    # Replace infinities with NaN then fill
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)
    X_scaled = scaler.transform(X_all)
    
    # Predict
    log_pred = model.predict(X_scaled)
    daily_df['Chl_proxy'] = 10**log_pred - 1
    
    # Fuse with YSI fluorescence
    # Use night RFU if available, otherwise use mean
    daily_df['Chl_RFU_best'] = daily_df['Chl_RFU_night'].fillna(daily_df['Chl RFU_mean'])
    
    # Weight by proximity to anchor points
    daily_df['has_anchor'] = daily_df['Chl_from_algae'].notna().astype(float)
    daily_df['days_since_anchor'] = daily_df.groupby('Site ID (new)')['has_anchor'].transform(
        lambda x: x.expanding().sum()
    )
    
    # Simple weighting scheme
    daily_df['w_RFU'] = 1 / (1 + np.exp(-0.1 * (daily_df['days_since_anchor'] - 10)))
    daily_df['w_proxy'] = 1 - daily_df['w_RFU']
    
    # Fused estimate
    daily_df['Chl_fused'] = (
        daily_df['w_RFU'] * daily_df['Chl ug/L_mean'] + 
        daily_df['w_proxy'] * daily_df['Chl_proxy']
    )
    
    return daily_df

def compute_qc_flags(daily_df):
    """Step 5: Compute QC flags"""
    
    print("Computing QC flags...")
    
    # Flag 1: NPQ signature (if daytime RFU available)
    daily_df['NPQ_flag'] = False  # Placeholder
    
    # Flag 2: Model deviation (YSI < lower 95% band for 3+ days)
    daily_df['YSI_proxy_ratio'] = daily_df['Chl ug/L_mean'] / (daily_df['Chl_proxy'] + 1.0)
    
    # Only flag when both YSI and proxy have meaningful values
    daily_df['low_bias_flag'] = (
        (daily_df['YSI_proxy_ratio'] < 0.5) & 
        (daily_df['Chl_proxy'] > 5)  # Only flag when proxy suggests meaningful Chl
    ).rolling(window=3, min_periods=3, center=True).sum() >= 3
    
    # Flag 3: DO-Chl mismatch
    daily_df['DO_Chl_mismatch'] = (
        (daily_df['DO_diel'] > 2.0) &  # High DO amplitude
        (daily_df['Chl ug/L_mean'] < daily_df['Chl ug/L_mean'].quantile(0.3))  # Low Chl
    )
    
    # Flag 4: Anchor disagreement
    daily_df['anchor_error'] = np.abs(
        daily_df['Chl ug/L_mean'] - daily_df['Chl_from_algae']
    ) / (daily_df['Chl_from_algae'] + 1)
    daily_df['anchor_disagree'] = daily_df['anchor_error'] > 0.5
    
    # Summary flag
    daily_df['any_QC_flag'] = (
        daily_df['low_bias_flag'] | 
        daily_df['DO_Chl_mismatch'] | 
        daily_df['anchor_disagree'].fillna(False)
    )
    
    return daily_df

def plot_results(daily_df, site='BCR'):
    """Simple visualization"""
    
    site_data = daily_df[daily_df['Site ID (new)'] == site].copy()
    site_data = site_data.sort_values('Date')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top: Chlorophyll comparison
    ax1 = axes[0]
    ax1.plot(site_data['Date'], site_data['Chl ug/L_mean'], 
             'b-', label='YSI Chl', alpha=0.6)
    ax1.plot(site_data['Date'], site_data['Chl_proxy'], 
             'r--', label='Proxy Model', alpha=0.6)
    ax1.plot(site_data['Date'], site_data['Chl_fused'], 
             'g-', label='Fused', linewidth=2)
    
    # Add anchor points
    anchors = site_data[site_data['Chl_from_algae'].notna()]
    ax1.scatter(anchors['Date'], anchors['Chl_from_algae'], 
               color='black', s=30, label='Algae counts', zorder=5)
    
    # Mark QC flags
    flags = site_data[site_data['any_QC_flag']]
    if len(flags) > 0:
        ax1.scatter(flags['Date'], flags['Chl ug/L_mean'], 
                   color='red', marker='v', s=50, alpha=0.5, label='QC Flag')
    
    ax1.set_ylabel('Chlorophyll-a (µg/L)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Site {site}: Chlorophyll QC Analysis')
    
    # Bottom: QC metrics
    ax2 = axes[1]
    ax2.bar(site_data['Date'], site_data['DO_diel'], 
            color='lightblue', alpha=0.5, label='DO diel amplitude')
    
    ax2_twin = ax2.twinx()
    ax2_twin.plot(site_data['Date'], site_data['YSI_proxy_ratio'], 
                  'purple', label='YSI/Proxy ratio', alpha=0.7)
    ax2_twin.axhline(y=1.0, color='black', linestyle='--', alpha=0.3)
    ax2_twin.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    
    ax2.set_ylabel('DO Amplitude (mg/L)', color='blue')
    ax2_twin.set_ylabel('YSI/Proxy Ratio', color='purple')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    return fig

def main():
    """Run the complete analysis"""
    
    print("\n=== CHLOROPHYLL QC ANALYSIS ===\n")
    
    # Step 1: Load data
    ysi, algae, nutrients = prepare_data()
    
    # Step 2: Compute features
    daily = compute_daily_features(ysi, algae, nutrients)
    
    # Step 3: Train model
    model, scaler, features = train_proxy_model(daily)
    
    # Step 4: Apply and fuse
    daily = apply_model_and_fuse(daily, model, scaler, features)
    
    # Step 5: QC flags
    daily = compute_qc_flags(daily)
    
    # Save results
    print("\nSaving results...")
    daily.to_csv('chlorophyll_qc_results.csv', index=False)
    
    # Print summary
    print("\n=== QC SUMMARY ===")
    print(f"Total days analyzed: {len(daily)}")
    print(f"Days with QC flags: {daily['any_QC_flag'].sum()} ({100*daily['any_QC_flag'].mean():.1f}%)")
    print(f"Low bias events: {daily['low_bias_flag'].sum()}")
    print(f"DO-Chl mismatches: {daily['DO_Chl_mismatch'].sum()}")
    
    # Site-specific summary
    site_summary = daily.groupby('Site ID (new)')['any_QC_flag'].agg(['sum', 'mean'])
    site_summary.columns = ['QC_flags', 'QC_rate']
    site_summary['QC_rate'] = (site_summary['QC_rate'] * 100).round(1)
    print("\nQC rates by site:")
    print(site_summary)
    
    # Generate plot
    print("\nGenerating visualization...")
    fig = plot_results(daily, site='BCR')
    plt.savefig('chlorophyll_qc_plot.png', dpi=150, bbox_inches='tight')
    print("Saved: chlorophyll_qc_plot.png")
    
    print("\n✓ Analysis complete!")
    print("  Results: chlorophyll_qc_results.csv")
    print("  Plot: chlorophyll_qc_plot.png")
    
    return daily

if __name__ == "__main__":
    results = main()