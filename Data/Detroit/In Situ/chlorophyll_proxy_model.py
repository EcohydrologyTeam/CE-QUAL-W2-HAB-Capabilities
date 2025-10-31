"""
Chlorophyll Proxy Model for Detroit Lake
Integrates YSI fluorescence with algae counts and environmental proxies
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class ChlorophyllProxyModel:
    """
    Builds a multi-source chlorophyll estimation model using:
    - YSI fluorescence (primary high-frequency signal)
    - Algae enumeration (anchoring/calibration)
    - DO dynamics (GPP proxy)
    - Nutrients (when available)
    """
    
    def __init__(self, ysi_file, algae_file):
        self.ysi_file = ysi_file
        self.algae_file = algae_file
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and preprocess YSI and algae data"""
        
        # Load YSI data
        self.ysi = pd.read_excel(self.ysi_file)
        self.ysi['DateTime'] = pd.to_datetime(self.ysi['DateTime'])
        self.ysi['Date'] = self.ysi['DateTime'].dt.date
        
        # Load algae data
        self.algae = pd.read_excel(self.algae_file, sheet_name='AlgaeID_Enumeration')
        self.algae['Date'] = pd.to_datetime(self.algae['DATE']).dt.date
        
        # Load nutrients
        self.nutrients = pd.read_excel(self.algae_file, sheet_name='Nutrients_WillowLakeLab')
        self.nutrients['Date'] = pd.to_datetime(self.nutrients['Date'], errors='coerce').dt.date
        
        # Clean nutrient values (handle detection limits)
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
        
        for col in ['NH3-ISE (mg/L), lo-level', 'NO3+NO2 (mg/L)', 'OP-Phos (mg/L)']:
            if col in self.nutrients.columns:
                self.nutrients[col] = self.nutrients[col].apply(clean_nutrient_value)
        
        print(f"Loaded {len(self.ysi)} YSI records")
        print(f"Loaded {len(self.algae)} algae records")
        print(f"Loaded {len(self.nutrients)} nutrient records")
        
    def calculate_algae_metrics(self):
        """Calculate daily algae-based chlorophyll proxies"""
        
        # Aggregate algae counts by date and site
        algae_daily = self.algae.groupby(['Sample Site', 'Date']).agg({
            'DENSITY (cells/mL) REP 1': 'sum',
            'TOTAL BV (um3/mL)': 'sum',
            'DIVISION': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Calculate division proportions
        divisions = ['Cyanobacteria', 'Bacillariophyta', 'Chlorophyta', 'Cryptophyta']
        for div in divisions:
            algae_daily[f'prop_{div}'] = algae_daily['DIVISION'].apply(
                lambda x: x.get(div, 0) / sum(x.values()) if x else 0
            )
        
        # Estimate chlorophyll from biovolume and composition
        # Chl:biovolume ratios (pg Chl/µm³) - approximate values
        # Convert: um3/mL * pg/um3 * 1e-9 mg/pg * 1e3 µg/mg = um3/mL * 1e-6 = µg/L
        chl_ratios = {
            'Cyanobacteria': 3e-6,
            'Bacillariophyta': 5e-6,
            'Chlorophyta': 7e-6,
            'Cryptophyta': 6e-6
        }
        
        algae_daily['chl_from_algae'] = 0
        for div, ratio in chl_ratios.items():
            algae_daily['chl_from_algae'] += (
                algae_daily[f'prop_{div}'] * 
                algae_daily['TOTAL BV (um3/mL)'] * 
                ratio
            )
        
        # Log transform for modeling (handle NaNs)
        algae_daily['log_cells'] = np.log10(algae_daily['DENSITY (cells/mL) REP 1'].fillna(1) + 1)
        algae_daily['log_biovolume'] = np.log10(algae_daily['TOTAL BV (um3/mL)'].fillna(1) + 1)
        
        return algae_daily
    
    def calculate_do_metrics(self):
        """Calculate DO-based productivity metrics"""
        
        ysi_daily = self.ysi.groupby(['Site ID (new)', 'Date']).agg({
            'DO mg/L': ['mean', 'min', 'max', 'std'],
            'DO %': ['mean', 'min', 'max'],
            'Temp °C': 'mean',
            'pH': 'mean',
            'Chl ug/L': 'mean',
            'Chl RFU': 'mean',
            'BGA-PC RFU': 'mean'
        }).reset_index()
        
        # Flatten column names
        ysi_daily.columns = ['_'.join(col).strip('_') for col in ysi_daily.columns]
        
        # Calculate DO metrics
        ysi_daily['do_amplitude'] = ysi_daily['DO mg/L_max'] - ysi_daily['DO mg/L_min']
        ysi_daily['do_saturation_anomaly'] = ysi_daily['DO %_mean'] - 100
        
        # Calculate lagged features (1-7 days)
        lag_features = ['do_amplitude', 'do_saturation_anomaly', 'Temp °C_mean']
        for feature in lag_features:
            for lag in range(1, 8):
                ysi_daily[f'{feature}_lag{lag}'] = (
                    ysi_daily.groupby('Site ID (new)')[feature].shift(lag)
                )
        
        return ysi_daily
    
    def merge_datasets(self):
        """Merge YSI, algae, and nutrient data"""
        
        # Get processed datasets
        algae_metrics = self.calculate_algae_metrics()
        do_metrics = self.calculate_do_metrics()
        
        # Merge YSI with algae (anchor points)
        merged = do_metrics.merge(
            algae_metrics,
            left_on=['Site ID (new)', 'Date'],
            right_on=['Sample Site', 'Date'],
            how='left'
        )
        
        # Merge with nutrients
        nutrients_subset = self.nutrients[['Site Code', 'Date', 'NH3-ISE (mg/L), lo-level', 
                                          'NO3+NO2 (mg/L)', 'OP-Phos (mg/L)']].dropna(subset=['Date'])
        
        merged = merged.merge(
            nutrients_subset,
            left_on=['Site ID (new)', 'Date'],
            right_on=['Site Code', 'Date'],
            how='left',
            suffixes=('', '_nut')
        )
        
        # Log transform nutrients
        nut_cols = ['NH3-ISE (mg/L), lo-level', 'NO3+NO2 (mg/L)', 'OP-Phos (mg/L)']
        for col in nut_cols:
            if col in merged.columns:
                merged[f'log_{col}'] = np.log10(merged[col] + 0.001)
        
        return merged
    
    def build_features(self, df):
        """Build feature matrix for modeling"""
        
        feature_cols = [
            # DO metrics
            'do_amplitude', 'do_saturation_anomaly', 'DO mg/L_mean', 'DO %_mean',
            # Environmental
            'Temp °C_mean', 'pH_mean',
            # Fluorescence
            'Chl RFU_mean', 'BGA-PC RFU_mean',
            # Lagged DO features
            'do_amplitude_lag1', 'do_amplitude_lag3', 'do_amplitude_lag7',
            'do_saturation_anomaly_lag1', 'do_saturation_anomaly_lag3',
            # Nutrients (if available)
            'log_NH3-ISE (mg/L), lo-level', 'log_NO3+NO2 (mg/L)', 'log_OP-Phos (mg/L)',
            # Algae composition (if available)
            'prop_Cyanobacteria', 'prop_Bacillariophyta', 'prop_Chlorophyta'
        ]
        
        # Select available features
        available_features = [f for f in feature_cols if f in df.columns]
        X = df[available_features].copy()
        
        # Add seasonality
        df['day_of_year'] = pd.to_datetime(df['Date']).dt.dayofyear
        X['sin_doy'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        X['cos_doy'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return X
    
    def train_model(self, model_type='elastic_net'):
        """Train the proxy model using anchor points"""
        
        # Get merged data
        df = self.merge_datasets()
        
        # Filter to anchor points (where we have algae counts)
        anchor_df = df[df['chl_from_algae'].notna()].copy()
        
        print(f"\nTraining on {len(anchor_df)} anchor points")
        
        # Build features and target
        X = self.build_features(anchor_df)
        y = np.log10(anchor_df['chl_from_algae'] + 1)  # Log transform target
        
        # Remove rows with NaN features
        mask = X.notna().all(axis=1)
        X = X[mask]
        y = y[mask]
        
        # Replace infinities with NaN then fill
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model with time series cross-validation
        if model_type == 'elastic_net':
            self.model = ElasticNetCV(
                cv=TimeSeriesSplit(n_splits=5),
                alphas=np.logspace(-3, 1, 20),
                l1_ratio=[0.1, 0.5, 0.9],
                max_iter=5000
            )
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        self.model.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = self.model.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        print(f"Model performance on anchor points:")
        print(f"  R²: {r2:.3f}")
        print(f"  RMSE (log scale): {rmse:.3f}")
        
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        return self.model
    
    def predict_full_timeseries(self):
        """Generate predictions for all dates"""
        
        # Get merged data
        df = self.merge_datasets()
        
        # Build features for all dates
        X = self.build_features(df)
        
        # Ensure same features as training
        X = X[self.feature_names]
        
        # Handle missing values
        X_filled = X.ffill().bfill()
        
        # Replace infinities
        X_filled = X_filled.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_filled)
        log_pred = self.model.predict(X_scaled)
        
        # Back-transform
        df['chl_proxy'] = 10**log_pred - 1
        
        # Combine with YSI fluorescence using weighted average
        # Weight by temporal distance to nearest anchor point
        df['days_to_anchor'] = df.groupby('Site ID (new)')['chl_from_algae'].transform(
            lambda x: x.notna().rolling(window=30, center=True, min_periods=1).sum()
        )
        
        df['weight_ysi'] = 1 / (1 + df['days_to_anchor'])
        df['weight_proxy'] = 1 - df['weight_ysi']
        
        df['chl_fused'] = (
            df['weight_ysi'] * df['Chl ug/L_mean'] + 
            df['weight_proxy'] * df['chl_proxy']
        )
        
        return df
    
    def generate_qc_flags(self, df):
        """Generate QC flags for YSI sensor issues"""
        
        flags = []
        
        # Flag 1: YSI much lower than proxy (>50% deviation for 3+ days)
        df['ysi_proxy_ratio'] = df['Chl ug/L_mean'] / (df['chl_proxy'] + 1.0)
        df['low_bias_flag'] = (
            (df['ysi_proxy_ratio'] < 0.5) &
            (df['chl_proxy'] > 5)  # Only flag when proxy suggests meaningful Chl
        ).rolling(window=3, min_periods=3, center=True).sum() >= 3
        
        # Flag 2: High DO amplitude but low fluorescence
        df['do_chl_mismatch'] = (
            (df['do_amplitude'] > df['do_amplitude'].quantile(0.75)) &
            (df['Chl ug/L_mean'] < df['Chl ug/L_mean'].quantile(0.25))
        )
        
        # Flag 3: Sudden drops in fluorescence
        df['rfu_change'] = df.groupby('Site ID (new)')['Chl RFU_mean'].pct_change()
        df['sudden_drop_flag'] = df['rfu_change'] < -0.5
        
        return df
    
    def run_analysis(self):
        """Run complete analysis pipeline"""
        
        print("=== Chlorophyll Proxy Model Analysis ===\n")
        
        # Load data
        print("1. Loading data...")
        self.load_data()
        
        # Train model
        print("\n2. Training proxy model...")
        self.train_model(model_type='elastic_net')
        
        # Generate predictions
        print("\n3. Generating full time series...")
        results = self.predict_full_timeseries()
        
        # Add QC flags
        print("\n4. Computing QC flags...")
        results = self.generate_qc_flags(results)
        
        # Save results
        print("\n5. Saving results...")
        results.to_csv('chlorophyll_proxy_results.csv', index=False)
        
        # Print summary
        print("\n=== Summary ===")
        print(f"Total predictions: {len(results)}")
        print(f"Low bias flags: {results['low_bias_flag'].sum()}")
        print(f"DO-Chl mismatches: {results['do_chl_mismatch'].sum()}")
        print(f"Sudden drops: {results['sudden_drop_flag'].sum()}")
        
        return results


if __name__ == "__main__":
    # Initialize model
    model = ChlorophyllProxyModel(
        ysi_file='CityofSalem_YSI_RawData.xlsx',
        algae_file='CityofSalem_NutrientsAlgae_Raw.xlsx'
    )
    
    # Run analysis
    results = model.run_analysis()
    
    print("\nAnalysis complete! Results saved to 'chlorophyll_proxy_results.csv'")