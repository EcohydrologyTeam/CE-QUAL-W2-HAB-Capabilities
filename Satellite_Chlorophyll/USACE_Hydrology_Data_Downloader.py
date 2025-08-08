#!/usr/bin/env python3
"""
USACE Hydrology Data Downloader

Downloads hydrological data from US Army Corps of Engineers (USACE) 
Common Data Access (CDA) API for specified reservoirs and parameters.

Features:
- Download pool elevation, storage, inflow, and outflow data
- Support for multiple reservoirs and districts
- Date range specification (2001-present)
- CSV output format
- Automatic retry on failures

Author: Enhanced version
Date: 2025-08-01
"""

import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# USACE Districts and their API endpoints
DISTRICTS = {
    'SWT': 'swt',  # Tulsa District (Kansas, Oklahoma, Texas)
    'NWK': 'nwk',  # Kansas City District
    'NWO': 'nwo',  # Omaha District
    'NWP': 'nwp',  # Portland District
    'SWF': 'swf',  # Fort Worth District
    'SAC': 'sac',  # Sacramento District
    'SPK': 'spk',  # Sacramento District
}

# Reservoir configurations with their district codes
RESERVOIRS = {
    # Kansas reservoirs (Tulsa District)
    'Kanopolis': {'code': 'KANO', 'district': 'swt', 'state': 'KS'},
    'Wilson': {'code': 'WILS', 'district': 'swt', 'state': 'KS'},
    'Marion': {'code': 'MARI', 'district': 'swt', 'state': 'KS'},
    'Council Grove': {'code': 'COGR', 'district': 'swt', 'state': 'KS'},
    'John Redmond': {'code': 'JORE', 'district': 'swt', 'state': 'KS'},
    'Toronto': {'code': 'TORO', 'district': 'swt', 'state': 'KS'},
    'Fall River': {'code': 'FARE', 'district': 'swt', 'state': 'KS'},
    'Elk City': {'code': 'ELKC', 'district': 'swt', 'state': 'KS'},
    'Big Hill': {'code': 'BIHI', 'district': 'swt', 'state': 'KS'},
    
    # Add more reservoirs as needed
    'Detroit': {'code': 'DET', 'district': 'nwp', 'state': 'OR'},
}

# Parameter configurations
PARAMETERS = {
    'Elev': {
        'name_template': '{code}.Elev.1Hour.0.Best-{DISTRICT}',
        'units': 'ft',
        'description': 'Pool Elevation'
    },
    'Stor': {
        'name_template': '{code}.Stor.1Hour.0.Best-{DISTRICT}',
        'units': 'ac-ft',
        'description': 'Storage Volume'
    },
    'Flow-In': {
        'name_template': '{code}.Flow-In.1Hour.0.Best-{DISTRICT}',
        'units': 'cfs',
        'description': 'Inflow'
    },
    'Flow-Out': {
        'name_template': '{code}.Flow-Out.1Hour.0.Best-{DISTRICT}',
        'units': 'cfs',
        'description': 'Outflow'
    },
}


class USACEDataDownloader:
    """Class to handle USACE data downloads."""
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the downloader.
        
        Args:
            verbose: Print detailed progress messages
        """
        self.verbose = verbose
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'USACE-Data-Downloader/1.0',
            'Accept': 'application/json'
        })
    
    def _print(self, message: str):
        """Print message if verbose mode is enabled."""
        if self.verbose:
            print(message)
    
    def download_data(
        self,
        reservoir: str,
        parameter: str,
        start_date: str,
        end_date: str,
        district: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Download data for a specific reservoir and parameter.
        
        Args:
            reservoir: Reservoir name or code
            parameter: Parameter type (Elev, Stor, Flow-In, Flow-Out)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            district: Optional district override
        
        Returns:
            DataFrame with downloaded data or None if failed
        """
        # Get reservoir info
        if reservoir in RESERVOIRS:
            res_info = RESERVOIRS[reservoir]
            res_code = res_info['code']
            res_district = district or res_info['district']
        else:
            # Assume it's a code directly
            res_code = reservoir.upper()
            res_district = district or 'swt'
        
        # Get parameter info
        if parameter not in PARAMETERS:
            self._print(f"Error: Unknown parameter '{parameter}'")
            return None
        
        param_info = PARAMETERS[parameter]
        
        # Build time series name
        ts_name = param_info['name_template'].format(
            code=res_code,
            DISTRICT=res_district.upper()
        )
        
        # Build API URL
        base_url = f"https://water.usace.army.mil/cda/reporting/providers/{res_district}/timeseries"
        
        # Format dates for API
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # API requires ISO format with timezone
        params = {
            'name': ts_name,
            'begin': start_dt.strftime('%Y-%m-%dT00:00:00.000Z'),
            'end': end_dt.strftime('%Y-%m-%dT23:59:59.000Z')
        }
        
        self._print(f"\nDownloading data for {res_code} - {parameter}")
        self._print(f"URL: {base_url}")
        self._print(f"Parameters: {params}")
        
        # Make request with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.session.get(base_url, params=params, timeout=30)
                
                if response.status_code == 200:
                    # Check if response has content
                    if not response.content:
                        self._print(f"Warning: Empty response for {res_code} - {parameter}")
                        return None
                    
                    # Parse JSON response
                    data = response.json()
                    
                    # Convert to DataFrame
                    if 'values' in data:
                        df = self._parse_response(data, res_code, parameter)
                        if df is not None and not df.empty:
                            self._print(f"Successfully downloaded {len(df)} records")
                            return df
                        else:
                            self._print(f"No data available for specified date range")
                            return None
                    else:
                        self._print(f"Unexpected response format: {data.keys()}")
                        return None
                        
                elif response.status_code == 404:
                    self._print(f"Error: Time series '{ts_name}' not found")
                    return None
                else:
                    self._print(f"Error: HTTP {response.status_code} - {response.text[:200]}")
                    
            except json.JSONDecodeError as e:
                self._print(f"Error parsing JSON response: {e}")
                self._print(f"Response content: {response.text[:500]}")
            except requests.exceptions.RequestException as e:
                self._print(f"Request error (attempt {attempt + 1}/{max_retries}): {e}")
            except Exception as e:
                self._print(f"Unexpected error: {e}")
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
        
        self._print(f"Failed to download data for {reservoir} - {parameter}")
        return None
    
    def _parse_response(self, data: dict, res_code: str, parameter: str) -> Optional[pd.DataFrame]:
        """
        Parse USACE API response into DataFrame.
        
        Args:
            data: JSON response data
            res_code: Reservoir code
            parameter: Parameter name
        
        Returns:
            Parsed DataFrame or None
        """
        try:
            # Extract values
            values = data.get('values', [])
            if not values:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(values)
            
            # Expected columns: 'time' and 'value'
            if 'time' not in df.columns or 'value' not in df.columns:
                self._print(f"Warning: Unexpected columns in response: {df.columns.tolist()}")
                return None
            
            # Convert time to datetime
            df['datetime'] = pd.to_datetime(df['time'])
            
            # Rename value column
            param_col = f"{res_code}_{parameter}"
            df = df.rename(columns={'value': param_col})
            
            # Add metadata
            df['reservoir'] = res_code
            df['parameter'] = parameter
            df['units'] = PARAMETERS[parameter]['units']
            
            # Select and reorder columns
            columns = ['datetime', 'reservoir', 'parameter', param_col, 'units']
            df = df[columns]
            
            # Sort by datetime
            df = df.sort_values('datetime')
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['datetime'])
            
            return df
            
        except Exception as e:
            self._print(f"Error parsing response: {e}")
            return None
    
    def download_multiple(
        self,
        reservoirs: List[str],
        parameters: List[str],
        start_date: str,
        end_date: str,
        output_dir: str = '.'
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple reservoirs and parameters.
        
        Args:
            reservoirs: List of reservoir names
            parameters: List of parameter types
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            output_dir: Directory to save CSV files
        
        Returns:
            Dictionary of DataFrames by reservoir-parameter combination
        """
        results = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for reservoir in reservoirs:
            for parameter in parameters:
                key = f"{reservoir}_{parameter}"
                
                # Download data
                df = self.download_data(
                    reservoir=reservoir,
                    parameter=parameter,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if df is not None and not df.empty:
                    results[key] = df
                    
                    # Save to CSV
                    filename = output_path / f"{key}_{start_date}_to_{end_date}.csv"
                    df.to_csv(filename, index=False)
                    self._print(f"Saved data to {filename}")
                else:
                    self._print(f"No data available for {key}")
                
                # Brief pause between requests
                time.sleep(0.5)
        
        return results


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Download USACE hydrological data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download elevation data for Kanopolis reservoir for 2024
  python USACE_Hydrology_Data_Downloader.py --reservoirs Kanopolis --parameters Elev --start 2024-01-01 --end 2024-12-31
  
  # Download all parameters for multiple reservoirs
  python USACE_Hydrology_Data_Downloader.py --reservoirs Kanopolis Wilson --parameters Elev Stor Flow-In Flow-Out --start 2001-01-01
  
  # Run example download (last 30 days)
  python USACE_Hydrology_Data_Downloader.py --example
        """
    )
    
    parser.add_argument(
        '--reservoirs',
        nargs='+',
        help='Reservoir names (e.g., Kanopolis Wilson)'
    )
    parser.add_argument(
        '--parameters',
        nargs='+',
        choices=['Elev', 'Stor', 'Flow-In', 'Flow-Out'],
        help='Parameters to download'
    )
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD). Default: 2001-01-01'
    )
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory for CSV files (default: current directory)'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='Run example download (last 30 days of elevation data)'
    )
    parser.add_argument(
        '--list-reservoirs',
        action='store_true',
        help='List available reservoirs'
    )
    
    args = parser.parse_args()
    
    # List reservoirs if requested
    if args.list_reservoirs:
        print("\nAvailable Reservoirs:")
        print("-" * 50)
        for name, info in RESERVOIRS.items():
            print(f"{name:20} Code: {info['code']:6} District: {info['district']:4} State: {info['state']}")
        sys.exit(0)
    
    # Set defaults
    today = datetime.now().strftime('%Y-%m-%d')
    
    if args.example:
        # Example: Last 30 days of elevation data
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        reservoirs = ['Kanopolis', 'Wilson']
        parameters = ['Elev']
        print(f"Downloading pool elevation data from {start_date} to {end_date}")
        print(f"Reservoirs: {', '.join(reservoirs)}")
    else:
        # Use provided arguments or defaults
        reservoirs = args.reservoirs or ['Kanopolis']
        parameters = args.parameters or ['Elev']
        start_date = args.start or '2001-01-01'
        end_date = args.end or today
        
        # Validate dates
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            if start_dt > end_dt:
                print("Error: Start date must be before end date")
                sys.exit(1)
            
            if end_dt > datetime.now():
                print(f"Warning: End date {end_date} is in the future. Using today's date instead.")
                end_date = today
                
        except ValueError as e:
            print(f"Error: Invalid date format. Use YYYY-MM-DD. {e}")
            sys.exit(1)
    
    # Create downloader and run
    print("-" * 50)
    downloader = USACEDataDownloader(verbose=True)
    
    results = downloader.download_multiple(
        reservoirs=reservoirs,
        parameters=parameters,
        start_date=start_date,
        end_date=end_date,
        output_dir=args.output
    )
    
    # Summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Successful downloads: {len(results)}")
    
    if results:
        print("\nDatasets downloaded:")
        for key, df in results.items():
            print(f"  {key}: {len(df)} records")
    else:
        print("\nNo data was successfully downloaded.")
        print("This could be due to:")
        print("  1. Invalid reservoir codes or parameter names")
        print("  2. No data available for the specified date range")
        print("  3. Network connectivity issues")
        print("\nTry running with --list-reservoirs to see available options")


if __name__ == '__main__':
    main()