"""
Explain what variables are used for chlorophyll prediction
"""

import pandas as pd
import numpy as np

# Load the results to explain
results = pd.read_csv('chlorophyll_qc_results.csv')

print("=" * 70)
print("CHLOROPHYLL PREDICTION MODEL - VARIABLE EXPLANATION")
print("=" * 70)

print("\n📊 PREDICTOR VARIABLES (Inputs to the model):")
print("-" * 50)

predictors = {
    'DO_diel': 'Daily DO amplitude (max - min), proxy for photosynthesis',
    'DO_sat_anomaly': 'DO saturation deviation from 100%, indicates productivity',
    'DO mg/L_mean': 'Average dissolved oxygen concentration',
    'Temp °C_mean': 'Water temperature',
    'pH_mean': 'pH level (increases with photosynthesis)',
    'DO_diel_lag1': 'Yesterday\'s DO amplitude',
    'DO_diel_lag3': '3-day lagged DO amplitude',
    'DO_diel_lag7': 'Week-ago DO amplitude',
    'sin_doy, cos_doy': 'Seasonal patterns (day of year)',
    'Chl_RFU_night': 'Night-time fluorescence (minimizes quenching)',
    'share_Cyanobacteria': 'Proportion of cyanobacteria in algae community',
    'share_Bacillariophyta': 'Proportion of diatoms in algae community',
    'log_NH3, log_NO3, log_PO4': 'Log-transformed nutrient concentrations'
}

for var, description in predictors.items():
    if 'log_' in var:
        print(f"\n  • {var}")
    else:
        print(f"\n  • {var}")
    print(f"    {description}")
    if var in results.columns:
        coverage = results[var].notna().sum() / len(results) * 100
        print(f"    Coverage: {coverage:.1f}% of days")

print("\n\n🎯 TARGET VARIABLE (What we're predicting):")
print("-" * 50)
print("\n  • Chlorophyll-a concentration (µg/L)")
print("    Trained on algae biovolume converted to chlorophyll")
print(f"    Training samples: {results['Chl_from_algae'].notna().sum()} days with algae counts")

print("\n\n🔬 MODEL APPROACH:")
print("-" * 50)
print("""
1. TRAINING PHASE:
   - Uses only days with algae counts (ground truth)
   - Learns relationship between DO dynamics and chlorophyll
   - Elastic Net regression with cross-validation

2. PREDICTION PHASE:
   - Applied to all days with YSI measurements
   - Uses DO patterns to estimate chlorophyll
   - Key insight: High DO amplitude = high photosynthesis = high chlorophyll

3. OUTPUT:
   - Chl_proxy: Model prediction based on environmental variables
   - Chl_fused: Weighted average of YSI and proxy
   - QC flags: Identifies when YSI likely underestimates
""")

print("\n\n📈 WHY THIS WORKS:")
print("-" * 50)
print("""
• Photosynthesis produces oxygen during daylight
• More algae = more photosynthesis = larger DO swings
• DO amplitude (daily max - min) correlates with algal biomass
• This relationship persists even when fluorescence sensors fail
• Temperature, pH, and nutrients modulate this relationship
""")

print("\n\n⚠️  DATA STRUCTURE:")
print("-" * 50)

# Analyze temporal structure
for site in ['BCR', 'LB', 'HT']:
    if site in results['Site ID (new)'].values:
        site_data = results[results['Site ID (new)'] == site]
        site_dates = pd.to_datetime(site_data['Date'])
        
        if len(site_dates) > 1:
            gaps = site_dates.sort_values().diff().dt.days.dropna()
            print(f"\nSite {site}:")
            print(f"  • {len(site_data)} measurement days")
            print(f"  • Typical gap: {gaps.median():.0f} days")
            print(f"  • Date range: {site_dates.min():%Y-%m-%d} to {site_dates.max():%Y-%m-%d}")

print("\n\n🔍 KEY FINDING:")
print("-" * 50)
print(f"""
The model reveals systematic YSI underestimation:
  • YSI average: {results['Chl ug/L_mean'].mean():.2f} µg/L
  • Model prediction: {results['Chl_proxy'].mean():.2f} µg/L
  • Algae-based estimate: {results['Chl_from_algae'].dropna().mean():.2f} µg/L

This ~10x discrepancy suggests sensor calibration issues.
""")