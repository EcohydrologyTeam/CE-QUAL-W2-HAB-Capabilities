"""
Visualization tools for chlorophyll proxy model QC
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

def plot_chlorophyll_comparison(results_df, site='BCR', start_date=None, end_date=None):
    """
    Plot YSI fluorescence vs proxy model with QC flags
    """
    
    # Filter by site and date range
    site_data = results_df[results_df['Site ID (new)'] == site].copy()
    site_data['Date'] = pd.to_datetime(site_data['Date'])
    
    if start_date:
        site_data = site_data[site_data['Date'] >= start_date]
    if end_date:
        site_data = site_data[site_data['Date'] <= end_date]
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: Chlorophyll comparison
    ax1 = axes[0]
    ax1.plot(site_data['Date'], site_data['Chl ug/L_mean'], 
             'b-', label='YSI Fluorescence', alpha=0.7)
    ax1.plot(site_data['Date'], site_data['chl_proxy'], 
             'r--', label='Proxy Model', alpha=0.7)
    ax1.plot(site_data['Date'], site_data['chl_fused'], 
             'g-', label='Fused Estimate', linewidth=2)
    
    # Add anchor points
    anchor_data = site_data[site_data['chl_from_algae'].notna()]
    ax1.scatter(anchor_data['Date'], anchor_data['chl_from_algae'], 
               color='black', s=50, marker='o', label='Algae Counts', zorder=5)
    
    # Highlight QC flags
    flags = site_data[site_data['low_bias_flag']]
    if len(flags) > 0:
        ax1.scatter(flags['Date'], flags['Chl ug/L_mean'], 
                   color='red', s=100, marker='v', label='Low Bias Flag', alpha=0.5)
    
    ax1.set_ylabel('Chlorophyll-a (µg/L)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Chlorophyll-a Time Series - Site {site}')
    
    # Plot 2: DO metrics
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    
    ax2.bar(site_data['Date'], site_data['do_amplitude'], 
            color='lightblue', alpha=0.5, label='DO Amplitude')
    ax2_twin.plot(site_data['Date'], site_data['do_saturation_anomaly'], 
                  'orange', label='DO Saturation Anomaly', linewidth=1.5)
    
    ax2.set_ylabel('DO Amplitude (mg/L)', color='blue')
    ax2_twin.set_ylabel('DO Saturation Anomaly (%)', color='orange')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Dissolved Oxygen Dynamics')
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 3: Model confidence
    ax3 = axes[2]
    
    # Calculate rolling statistics for confidence bands  
    window = 7
    site_data['chl_rolling_mean'] = site_data['chl_fused'].rolling(window=window, center=True).mean()
    site_data['chl_rolling_std'] = site_data['chl_fused'].rolling(window=window, center=True).std()
    
    ax3.plot(site_data['Date'], site_data['ysi_proxy_ratio'], 
             'purple', label='YSI/Proxy Ratio', alpha=0.7)
    ax3.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Low Bias Threshold')
    ax3.axhline(y=2.0, color='orange', linestyle='--', alpha=0.5, label='High Bias Threshold')
    
    ax3.fill_between([site_data['Date'].min(), site_data['Date'].max()], 
                     0.5, 2.0, alpha=0.1, color='green', label='Acceptable Range')
    
    ax3.set_ylabel('YSI/Proxy Ratio')
    ax3.set_xlabel('Date')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Sensor QC Metrics')
    ax3.set_ylim([0, 3])
    
    plt.tight_layout()
    return fig

def plot_site_comparison(results_df):
    """
    Compare chlorophyll patterns across all sites
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    sites = results_df['Site ID (new)'].unique()[:9]  # Plot up to 9 sites
    
    for idx, site in enumerate(sites):
        ax = axes[idx]
        site_data = results_df[results_df['Site ID (new)'] == site].copy()
        site_data['Date'] = pd.to_datetime(site_data['Date'])
        
        # Resample to weekly for cleaner visualization
        site_weekly = site_data.set_index('Date').resample('W').agg({
            'Chl ug/L_mean': 'mean',
            'chl_proxy': 'mean',
            'chl_fused': 'mean',
            'chl_from_algae': 'mean'
        }).reset_index()
        
        ax.plot(site_weekly['Date'], site_weekly['Chl ug/L_mean'], 
                'b-', label='YSI', alpha=0.6, linewidth=1)
        ax.plot(site_weekly['Date'], site_weekly['chl_proxy'], 
                'r--', label='Proxy', alpha=0.6, linewidth=1)
        ax.plot(site_weekly['Date'], site_weekly['chl_fused'], 
                'g-', label='Fused', linewidth=1.5)
        
        # Add anchor points
        anchor_data = site_weekly[site_weekly['chl_from_algae'].notna()]
        if len(anchor_data) > 0:
            ax.scatter(anchor_data['Date'], anchor_data['chl_from_algae'], 
                      color='black', s=20, marker='o', alpha=0.5)
        
        ax.set_title(f'Site {site}')
        ax.set_xlabel('')
        ax.set_ylabel('Chl-a (µg/L)')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        if idx == 0:
            ax.legend(loc='upper left', fontsize=8)
    
    # Remove unused subplots
    for idx in range(len(sites), 9):
        fig.delaxes(axes[idx])
    
    plt.suptitle('Chlorophyll-a Comparison Across Sites', fontsize=14)
    plt.tight_layout()
    return fig

def generate_qc_report(results_df):
    """
    Generate a QC summary report
    """
    
    print("=" * 60)
    print("CHLOROPHYLL SENSOR QC REPORT")
    print("=" * 60)
    
    # Overall statistics
    print("\n1. OVERALL STATISTICS")
    print("-" * 40)
    print(f"Total measurements: {len(results_df):,}")
    print(f"Date range: {results_df['Date'].min()} to {results_df['Date'].max()}")
    print(f"Sites monitored: {results_df['Site ID (new)'].nunique()}")
    
    # Anchor point coverage
    anchor_coverage = results_df['chl_from_algae'].notna().sum()
    print(f"\nAnchor points (algae counts): {anchor_coverage:,} ({100*anchor_coverage/len(results_df):.1f}%)")
    
    # QC flag summary
    print("\n2. QC FLAG SUMMARY")
    print("-" * 40)
    
    flags = {
        'Low Bias (YSI < 50% of proxy)': results_df['low_bias_flag'].sum(),
        'DO-Chl Mismatch': results_df['do_chl_mismatch'].sum(),
        'Sudden Drops (>50% decrease)': results_df['sudden_drop_flag'].sum()
    }
    
    for flag_name, count in flags.items():
        pct = 100 * count / len(results_df)
        print(f"{flag_name}: {count:,} ({pct:.1f}%)")
    
    # Site-specific issues
    print("\n3. SITE-SPECIFIC QC ISSUES")
    print("-" * 40)
    
    site_summary = results_df.groupby('Site ID (new)').agg({
        'low_bias_flag': 'sum',
        'do_chl_mismatch': 'sum',
        'sudden_drop_flag': 'sum',
        'chl_from_algae': lambda x: x.notna().sum()
    }).rename(columns={'chl_from_algae': 'anchor_points'})
    
    site_summary['total_flags'] = (
        site_summary['low_bias_flag'] + 
        site_summary['do_chl_mismatch'] + 
        site_summary['sudden_drop_flag']
    )
    
    site_summary = site_summary.sort_values('total_flags', ascending=False)
    print("\nSites with most QC issues:")
    print(site_summary.head())
    
    # Time periods of concern
    print("\n4. TEMPORAL PATTERNS")
    print("-" * 40)
    
    results_df['Month'] = pd.to_datetime(results_df['Date']).dt.month
    monthly_flags = results_df.groupby('Month')['low_bias_flag'].mean() * 100
    
    print("\nMonthly low bias frequency (%):")
    for month, pct in monthly_flags.items():
        month_name = pd.Timestamp(2020, month, 1).strftime('%B')
        print(f"  {month_name:12} {pct:5.1f}%")
    
    # Recommendations
    print("\n5. RECOMMENDATIONS")
    print("-" * 40)
    
    recommendations = []
    
    if flags['Low Bias (YSI < 50% of proxy)'] > len(results_df) * 0.05:
        recommendations.append("• High frequency of low bias flags (>5%) suggests sensor fouling or calibration drift")
        recommendations.append("  → Consider more frequent sensor cleaning/calibration")
    
    if flags['DO-Chl Mismatch'] > len(results_df) * 0.03:
        recommendations.append("• Frequent DO-Chl mismatches indicate potential fluorescence quenching")
        recommendations.append("  → Consider sampling during early morning hours to minimize NPQ effects")
    
    if site_summary['total_flags'].max() > 50:
        problem_site = site_summary.index[0]
        recommendations.append(f"• Site {problem_site} shows excessive QC issues")
        recommendations.append("  → Investigate site-specific factors (turbidity, fouling organisms)")
    
    if len(recommendations) == 0:
        recommendations.append("• Sensor performance appears satisfactory")
        recommendations.append("• Continue current maintenance schedule")
    
    for rec in recommendations:
        print(rec)
    
    print("\n" + "=" * 60)
    
    return site_summary

if __name__ == "__main__":
    # Load results
    print("Loading results...")
    results = pd.read_csv('chlorophyll_proxy_results.csv')
    
    # Generate QC report
    site_summary = generate_qc_report(results)
    
    # Create visualizations
    print("\nGenerating visualizations...")
    
    # Plot for a specific site
    fig1 = plot_chlorophyll_comparison(results, site='BCR', 
                                       start_date='2018-06-01', 
                                       end_date='2019-10-01')
    plt.savefig('chlorophyll_qc_timeseries.png', dpi=150, bbox_inches='tight')
    print("Saved: chlorophyll_qc_timeseries.png")
    
    # Plot site comparison
    fig2 = plot_site_comparison(results)
    plt.savefig('chlorophyll_site_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: chlorophyll_site_comparison.png")
    
    plt.show()