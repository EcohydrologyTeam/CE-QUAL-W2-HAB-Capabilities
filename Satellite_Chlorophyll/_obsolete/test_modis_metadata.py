"""
Test script to explore MODIS metadata and find time information
================================================================

This script examines MODIS images from Google Earth Engine to find
all available metadata properties that might contain timing information.
"""

import ee
import json
from datetime import datetime
import pandas as pd

# Initialize Earth Engine
print("Initializing Google Earth Engine...")
ee.Authenticate()
ee.Initialize(project='ee-toddsteissberg')  # Replace with your project ID

# Test location (Upper Klamath Lake)
test_point = ee.Geometry.Point([-121.900, 42.400])
test_date = '2024-08-01'

# MODIS collections
TERRA_COL = 'MODIS/061/MOD09GA'
AQUA_COL = 'MODIS/061/MYD09GA'

def explore_image_metadata(collection_id, sensor_name):
    """Explore all metadata properties of a MODIS image"""
    print(f"\n{'='*60}")
    print(f"Exploring {sensor_name} metadata")
    print('='*60)
    
    # Get a single image
    image = (ee.ImageCollection(collection_id)
             .filterDate(test_date, ee.Date(test_date).advance(5, 'day'))
             .filterBounds(test_point)
             .first())
    
    # Get image info
    try:
        info = image.getInfo()
        
        if info is None:
            print(f"No {sensor_name} images found for the test date")
            return None
            
        print(f"\nImage ID: {info.get('id', 'N/A')}")
        
        # Extract properties
        properties = info.get('properties', {})
        
        print(f"\nAll properties ({len(properties)} total):")
        print("-" * 40)
        
        # Sort properties by key for easier reading
        for key in sorted(properties.keys()):
            value = properties[key]
            # Truncate long values for display
            if isinstance(value, str) and len(value) > 60:
                value = value[:60] + "..."
            print(f"  {key}: {value}")
        
        # Look specifically for time-related properties
        print("\n" + "-" * 40)
        print("Time-related properties:")
        print("-" * 40)
        
        time_keywords = ['time', 'TIME', 'date', 'DATE', 'hour', 'HOUR', 
                        'minute', 'MINUTE', 'acquisition', 'ACQUISITION',
                        'SENSING', 'sensing', 'RANGE', 'PRODUCTION']
        
        time_props = {}
        for key, value in properties.items():
            if any(keyword in key for keyword in time_keywords):
                time_props[key] = value
                print(f"  {key}: {value}")
        
        if not time_props:
            print("  No obvious time-related properties found")
        
        # Check system properties
        print("\n" + "-" * 40)
        print("System properties:")
        print("-" * 40)
        
        system_props = {k: v for k, v in properties.items() if k.startswith('system:')}
        for key, value in sorted(system_props.items()):
            if key == 'system:time_start' or key == 'system:time_end':
                # Convert milliseconds to readable datetime
                try:
                    dt = datetime.fromtimestamp(value / 1000)
                    print(f"  {key}: {value} ({dt})")
                except:
                    print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value}")
        
        # Check if system:index contains time info
        system_index = properties.get('system:index', '')
        print(f"\nAnalyzing system:index: {system_index}")
        
        # MODIS system:index format might be like: MOD09GA_A2024214_...
        # Where 2024214 = Year 2024, Day of Year 214
        if system_index:
            parts = system_index.split('_')
            print(f"  Index parts: {parts}")
            
            # Look for date components in the index
            for part in parts:
                if part.startswith('A') and len(part) == 8:  # Like A2024214
                    try:
                        year = int(part[1:5])
                        doy = int(part[5:8])
                        print(f"  Found date encoding: Year={year}, DOY={doy}")
                    except:
                        pass
        
        # Try to get granule-specific metadata
        print("\n" + "-" * 40)
        print("Looking for granule/tile information:")
        print("-" * 40)
        
        granule_keys = ['GRINGPOINTLATITUDE', 'GRINGPOINTLONGITUDE', 
                       'GRANULEID', 'LOCALGRANULEID', 'PRODUCTIONDATETIME',
                       'RANGEBEGINNINGDATE', 'RANGEBEGINNINGTIME',
                       'RANGEENDINGDATE', 'RANGEENDINGTIME']
        
        for key in granule_keys:
            if key in properties:
                print(f"  {key}: {properties[key]}")
        
        return properties
        
    except Exception as e:
        print(f"Error getting image info: {e}")
        return None

def test_computed_properties(collection_id, sensor_name):
    """Test if we can compute or extract time from image properties"""
    print(f"\n{'='*60}")
    print(f"Testing computed properties for {sensor_name}")
    print('='*60)
    
    # Get a small collection
    collection = (ee.ImageCollection(collection_id)
                  .filterDate('2024-08-01', '2024-08-10')
                  .filterBounds(test_point)
                  .limit(5))
    
    # Try to extract various time formats
    def extract_time_info(img):
        # Get basic properties
        props = ee.Dictionary({
            'system_index': img.get('system:index'),
            'system_time_start': img.get('system:time_start'),
            'id': img.id(),
        })
        
        # Try to format time in different ways
        time_start = ee.Date(img.get('system:time_start'))
        
        props = props.combine({
            'date_ymd': time_start.format('YYYY-MM-dd'),
            'time_hms': time_start.format('HH:mm:ss'),
            'datetime': time_start.format('YYYY-MM-dd HH:mm:ss'),
            'hour': time_start.get('hour'),
            'minute': time_start.get('minute'),
            'second': time_start.get('second'),
            'millis': time_start.millis(),
        })
        
        return ee.Feature(None, props)
    
    # Convert to feature collection and get info
    features = collection.map(extract_time_info)
    
    try:
        fc_info = features.getInfo()
        
        if fc_info and 'features' in fc_info:
            print(f"\nFound {len(fc_info['features'])} images")
            
            # Convert to dataframe for easier viewing
            records = [f['properties'] for f in fc_info['features']]
            df = pd.DataFrame(records)
            
            if not df.empty:
                print("\nExtracted time information:")
                print(df[['system_index', 'date_ymd', 'time_hms', 'hour', 'minute', 'second']].to_string())
                
                # Check if all times are 00:00:00
                if 'time_hms' in df.columns:
                    unique_times = df['time_hms'].unique()
                    print(f"\nUnique times found: {unique_times}")
                    
                    if len(unique_times) == 1 and unique_times[0] == '00:00:00':
                        print("\n⚠️  All times are 00:00:00 - actual overpass time not in metadata")
                        print("\nSuggested solution: Use approximate overpass times:")
                        print(f"  - {sensor_name} Terra: ~10:30 AM local solar time")
                        print(f"  - {sensor_name} Aqua: ~1:30 PM local solar time")
            
            return df
        else:
            print("No features found")
            return None
            
    except Exception as e:
        print(f"Error extracting time info: {e}")
        return None

def check_band_metadata():
    """Check if any bands contain time metadata"""
    print(f"\n{'='*60}")
    print("Checking band-level metadata")
    print('='*60)
    
    # Get a single Terra image
    image = (ee.ImageCollection(TERRA_COL)
             .filterDate(test_date, ee.Date(test_date).advance(5, 'day'))
             .filterBounds(test_point)
             .first())
    
    try:
        # Get band names
        band_names = image.bandNames().getInfo()
        print(f"\nAvailable bands: {band_names[:5]}...")  # Show first 5
        
        # Check if there are any time-related bands
        time_bands = [b for b in band_names if 'time' in b.lower() or 'hour' in b.lower()]
        if time_bands:
            print(f"\nPotential time-related bands: {time_bands}")
        else:
            print("\nNo obvious time-related bands found")
            
    except Exception as e:
        print(f"Error checking bands: {e}")

# Main execution
if __name__ == "__main__":
    print("MODIS Metadata Exploration Test")
    print("=" * 60)
    print(f"Test location: Upper Klamath Lake ({-121.900}, {42.400})")
    print(f"Test date: {test_date}")
    
    # Test Terra
    terra_props = explore_image_metadata(TERRA_COL, 'MODIS Terra')
    terra_df = test_computed_properties(TERRA_COL, 'MODIS Terra')
    
    # Test Aqua
    aqua_props = explore_image_metadata(AQUA_COL, 'MODIS Aqua')
    aqua_df = test_computed_properties(AQUA_COL, 'MODIS Aqua')
    
    # Check band metadata
    check_band_metadata()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if terra_df is not None and not terra_df.empty:
        if 'time_hms' in terra_df.columns and all(terra_df['time_hms'] == '00:00:00'):
            print("\n❌ Actual overpass times NOT available in GEE metadata")
            print("\n✅ Recommended solution:")
            print("   Use approximate local solar times based on MODIS orbit:")
            print("   - Terra: ~10:30 AM local solar time")
            print("   - Aqua: ~1:30 PM local solar time")
            print("\n   For Oregon (UTC-8 or UTC-7):")
            print("   - Terra: ~18:30 UTC (10:30 AM PST) or ~17:30 UTC (10:30 AM PDT)")
            print("   - Aqua: ~21:30 UTC (1:30 PM PST) or ~20:30 UTC (1:30 PM PDT)")
        else:
            print("\n✅ Time information found in metadata!")
    
    print("\nTest completed.")