"""
Examine the Detroit Lake in situ data structure
"""

import pandas as pd
import numpy as np

def examine_detroit_insitu_data():
    """Examine the structure of Detroit Lake YSI data"""
    
    excel_path = '/Users/todd/GitHub/ecohydrology/CE-QUAL-W2-HAB-Capabilities/Data/Detroit/In Situ/CityofSalem_YSI_RawData.xlsx'
    
    try:
        # Read Excel file and examine sheets
        excel_file = pd.ExcelFile(excel_path)
        print("Available sheets:")
        for sheet in excel_file.sheet_names:
            print(f"  - {sheet}")
        
        # Read the first sheet to understand structure
        if excel_file.sheet_names:
            first_sheet = excel_file.sheet_names[0]
            print(f"\nExamining sheet: {first_sheet}")
            
            df = pd.read_excel(excel_path, sheet_name=first_sheet, nrows=20)
            
            print(f"\nColumns ({len(df.columns)}):")
            for i, col in enumerate(df.columns):
                print(f"  {i}: {col}")
            
            print(f"\nFirst 5 rows:")
            print(df.head())
            
            print(f"\nData types:")
            print(df.dtypes)
            
            # Look for chlorophyll-related columns
            chl_columns = [col for col in df.columns if 'chl' in col.lower() or 'chlor' in col.lower()]
            if chl_columns:
                print(f"\nChlorophyll-related columns:")
                for col in chl_columns:
                    print(f"  - {col}")
            
            # Look for date columns
            date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
            if date_columns:
                print(f"\nDate/time columns:")
                for col in date_columns:
                    print(f"  - {col}")
            
            # Look for location columns
            location_columns = [col for col in df.columns if any(x in col.lower() for x in ['lat', 'lon', 'location', 'site', 'station'])]
            if location_columns:
                print(f"\nLocation columns:")
                for col in location_columns:
                    print(f"  - {col}")
        
        # Try other sheets if available
        if len(excel_file.sheet_names) > 1:
            print(f"\n" + "="*50)
            print("Examining other sheets:")
            
            for sheet_name in excel_file.sheet_names[1:]:
                print(f"\nSheet: {sheet_name}")
                try:
                    sheet_df = pd.read_excel(excel_path, sheet_name=sheet_name, nrows=5)
                    print(f"  Shape: {sheet_df.shape}")
                    print(f"  Columns: {list(sheet_df.columns)}")
                except Exception as e:
                    print(f"  Error reading sheet: {e}")
                    
    except Exception as e:
        print(f"Error reading Excel file: {e}")

if __name__ == "__main__":
    examine_detroit_insitu_data()