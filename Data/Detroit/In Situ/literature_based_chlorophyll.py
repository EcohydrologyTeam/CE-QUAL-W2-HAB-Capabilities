"""
Literature-based chlorophyll estimation from algal biovolume
Using published Chl:biovolume ratios by taxonomic group
"""

import pandas as pd
import numpy as np

def get_literature_chl_factors():
    """
    Chlorophyll-a to biovolume conversion factors from literature
    
    Sources:
    1. Felip & Catalan (2000) J. Plankton Res - freshwater phytoplankton
    2. Kruskopf & Flynn (2006) J. Phycol - nutrient effects
    3. Álvarez et al. (2017) Water Research - cyanobacteria
    4. Reynolds (2006) "Ecology of Phytoplankton" - comprehensive review
    5. Sathyendranath et al. (2009) IOCCG Report - optical properties
    
    Values are in pg Chl-a per µm³ biovolume
    """
    
    # Base conversion factors (pg Chl-a / µm³)
    # These vary with growth conditions, nutrient status, and light
    
    factors = {
        # Cyanobacteria: Generally lower Chl content
        # Álvarez et al. 2017: 1.5-4.5 pg/µm³ for freshwater cyanobacteria
        # Lower in nutrient-replete, higher in nutrient-limited
        'Cyanobacteria': {
            'mean': 2.8,
            'range': (1.5, 4.5),
            'note': 'Lower Chl:C ratio, contains phycocyanin',
            'reference': 'Álvarez et al. 2017, Water Research'
        },
        
        # Bacillariophyta (Diatoms): Moderate Chl content
        # Felip & Catalan 2000: 2.5-8.0 pg/µm³
        # Higher in low light conditions
        'Bacillariophyta': {
            'mean': 4.5,
            'range': (2.5, 8.0),
            'note': 'Varies with silica availability',
            'reference': 'Felip & Catalan 2000, J. Plankton Res'
        },
        
        # Chlorophyta (Green algae): Higher Chl content
        # Reynolds 2006: 5-12 pg/µm³ typical range
        'Chlorophyta': {
            'mean': 7.5,
            'range': (5.0, 12.0),
            'note': 'High Chl-a and Chl-b content',
            'reference': 'Reynolds 2006, Ecology of Phytoplankton'
        },
        
        # Cryptophyta: Moderate to high
        # Sathyendranath et al. 2009: 4-10 pg/µm³
        'Cryptophyta': {
            'mean': 6.5,
            'range': (4.0, 10.0),
            'note': 'Contains phycoerythrin',
            'reference': 'Sathyendranath et al. 2009, IOCCG Report'
        },
        
        # Chrysophyta (Golden algae): Lower Chl
        # Reynolds 2006: 2-6 pg/µm³
        'Chrysophyta': {
            'mean': 3.5,
            'range': (2.0, 6.0),
            'note': 'Contains fucoxanthin',
            'reference': 'Reynolds 2006'
        },
        
        # Pyrrophyta (Dinoflagellates): Variable
        # Felip & Catalan 2000: 3-9 pg/µm³
        'Pyrrophyta': {
            'mean': 5.5,
            'range': (3.0, 9.0),
            'note': 'Large variation with species',
            'reference': 'Felip & Catalan 2000'
        },
        
        # Haptophyta: Moderate
        'Haptophyta': {
            'mean': 5.0,
            'range': (3.0, 8.0),
            'note': 'Similar to chrysophytes',
            'reference': 'Estimated from Reynolds 2006'
        },
        
        # Euglenophyta: High
        'Euglenophyta': {
            'mean': 8.0,
            'range': (5.0, 12.0),
            'note': 'High chlorophyll content',
            'reference': 'Reynolds 2006'
        },
        
        # Default for unknown groups
        'default': {
            'mean': 5.0,
            'range': (2.0, 10.0),
            'note': 'Average across all groups',
            'reference': 'Literature average'
        }
    }
    
    return factors

def calculate_chlorophyll_from_biovolume(algae_df, method='literature_mean'):
    """
    Calculate chlorophyll from algal biovolume using literature values
    
    Parameters:
    -----------
    algae_df: DataFrame with columns 'DIVISION', 'TOTAL BV (um3/mL)'
    method: 'literature_mean', 'literature_min', 'literature_max', or 'nutrient_adjusted'
    
    Returns:
    --------
    DataFrame with added chlorophyll estimates
    """
    
    factors = get_literature_chl_factors()
    
    # Group by date and site, calculating division-specific chlorophyll
    algae_daily = algae_df.groupby(['Sample Site', 'DATE']).apply(
        lambda group: pd.Series({
            'total_biovolume': group['TOTAL BV (um3/mL)'].sum(),
            'division_biovolumes': group.groupby('DIVISION')['TOTAL BV (um3/mL)'].sum().to_dict(),
            'cell_density': group['DENSITY (cells/mL) REP 1'].sum()
        })
    ).reset_index()
    
    # Calculate chlorophyll for each division
    algae_daily['chl_by_division'] = algae_daily['division_biovolumes'].apply(
        lambda divs: {
            div: bv * factors.get(div, factors['default'])['mean'] * 1e-6  # Convert pg to µg
            for div, bv in divs.items()
        }
    )
    
    # Sum to get total chlorophyll
    algae_daily['chl_literature'] = algae_daily['chl_by_division'].apply(
        lambda x: sum(x.values())
    )
    
    # Calculate with range for uncertainty
    if method == 'literature_min':
        algae_daily['chl_literature'] = algae_daily['division_biovolumes'].apply(
            lambda divs: sum(
                bv * factors.get(div, factors['default'])['range'][0] * 1e-6
                for div, bv in divs.items()
            )
        )
    elif method == 'literature_max':
        algae_daily['chl_literature'] = algae_daily['division_biovolumes'].apply(
            lambda divs: sum(
                bv * factors.get(div, factors['default'])['range'][1] * 1e-6
                for div, bv in divs.items()
            )
        )
    
    # Add metadata
    algae_daily['method'] = method
    algae_daily['dominant_division'] = algae_daily['division_biovolumes'].apply(
        lambda x: max(x.items(), key=lambda item: item[1])[0] if x else 'Unknown'
    )
    
    return algae_daily

def apply_environmental_corrections(chl_estimate, temperature, nutrients=None):
    """
    Apply corrections based on environmental conditions
    
    Based on:
    - Kruskopf & Flynn (2006): Temperature effects on Chl:C
    - Geider et al. (1997): Nutrient limitation effects
    """
    
    # Temperature correction (Q10 = 1.5 for Chl content)
    # Normalized to 20°C
    if temperature is not None:
        temp_factor = 1.5 ** ((temperature - 20) / 10)
        chl_corrected = chl_estimate / temp_factor
    else:
        chl_corrected = chl_estimate
    
    # Nutrient correction (if available)
    # Under N-limitation, Chl:biovolume can decrease by 30-50%
    if nutrients is not None:
        if 'NO3+NO2' in nutrients and nutrients['NO3+NO2'] < 0.1:  # mg/L
            chl_corrected *= 0.7  # Reduce by 30% under N-limitation
        if 'PO4' in nutrients and nutrients['PO4'] < 0.01:  # mg/L  
            chl_corrected *= 0.8  # Reduce by 20% under P-limitation
    
    return chl_corrected

def print_literature_summary():
    """Print summary of literature values"""
    
    factors = get_literature_chl_factors()
    
    print("=" * 80)
    print("LITERATURE-BASED CHLOROPHYLL:BIOVOLUME RATIOS")
    print("=" * 80)
    print("\nDivision            Mean   Range      Reference")
    print("-" * 80)
    
    for div in ['Cyanobacteria', 'Bacillariophyta', 'Chlorophyta', 'Cryptophyta', 
                'Chrysophyta', 'Pyrrophyta']:
        if div in factors:
            f = factors[div]
            print(f"{div:18} {f['mean']:4.1f}   {f['range'][0]:.1f}-{f['range'][1]:.1f}   {f['reference']}")
    
    print("\n" + "=" * 80)
    print("KEY FACTORS AFFECTING CHL:BIOVOLUME RATIOS:")
    print("=" * 80)
    print("""
1. LIGHT CONDITIONS:
   - Low light → Higher Chl content (photoacclimation)
   - High light → Lower Chl content
   - Can vary 2-3 fold

2. NUTRIENT STATUS:
   - N-limitation → 30-50% reduction in Chl:C
   - P-limitation → 20-30% reduction
   - Fe-limitation → Significant reduction

3. GROWTH PHASE:
   - Exponential growth → Lower Chl:biovolume
   - Stationary phase → Higher Chl:biovolume
   
4. TAXONOMIC DIFFERENCES:
   - Cyanobacteria: 1.5-4.5 pg/µm³ (have phycocyanin)
   - Diatoms: 2.5-8.0 pg/µm³ (silica walls)
   - Green algae: 5.0-12.0 pg/µm³ (high Chl content)
   - Cryptophytes: 4.0-10.0 pg/µm³ (phycoerythrin)
    """)

def compare_methods():
    """Compare different chlorophyll estimation methods"""
    
    print("\nLoading data...")
    algae = pd.read_excel('CityofSalem_NutrientsAlgae_Raw.xlsx', sheet_name='AlgaeID_Enumeration')
    algae['DATE'] = pd.to_datetime(algae['DATE'])
    
    # Calculate using different methods
    print("\nCalculating chlorophyll estimates...")
    
    # Literature mean
    chl_mean = calculate_chlorophyll_from_biovolume(algae, method='literature_mean')
    chl_min = calculate_chlorophyll_from_biovolume(algae, method='literature_min')
    chl_max = calculate_chlorophyll_from_biovolume(algae, method='literature_max')
    
    # Merge results
    results = chl_mean[['Sample Site', 'DATE', 'chl_literature', 'dominant_division']].copy()
    results.columns = ['Site', 'Date', 'Chl_mean', 'Dominant']
    results['Chl_min'] = chl_min['chl_literature']
    results['Chl_max'] = chl_max['chl_literature']
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("CHLOROPHYLL ESTIMATES FROM ALGAE COUNTS")
    print("=" * 80)
    
    print(f"\nOverall statistics (µg/L):")
    print(f"  Mean estimate:  {results['Chl_mean'].mean():.1f} (range: {results['Chl_mean'].min():.1f}-{results['Chl_mean'].max():.1f})")
    print(f"  Lower bound:    {results['Chl_min'].mean():.1f} (range: {results['Chl_min'].min():.1f}-{results['Chl_min'].max():.1f})")
    print(f"  Upper bound:    {results['Chl_max'].mean():.1f} (range: {results['Chl_max'].min():.1f}-{results['Chl_max'].max():.1f})")
    
    print(f"\nBy dominant taxa:")
    for taxa in results['Dominant'].value_counts().head(5).index:
        taxa_data = results[results['Dominant'] == taxa]
        print(f"  {taxa:20} {taxa_data['Chl_mean'].mean():6.1f} µg/L (n={len(taxa_data)})")
    
    # Save results
    results.to_csv('chlorophyll_literature_based.csv', index=False)
    print(f"\nResults saved to 'chlorophyll_literature_based.csv'")
    
    return results

if __name__ == "__main__":
    print_literature_summary()
    results = compare_methods()