"""
Debug why Sentinel-2 data only starts in 2018 instead of 2015
"""

import ee
import pandas as pd

# Initialize GEE (you may need to authenticate first)
try:
    ee.Initialize(project='ee-toddsteissberg')
except:
    ee.Authenticate()
    ee.Initialize(project='ee-toddsteissberg')

def debug_sentinel_availability():
    """Debug Sentinel-2 data availability for Upper Klamath Lake"""
    
    print("Debugging Sentinel-2 data availability for Upper Klamath Lake...")
    print("="*70)
    
    # Upper Klamath Lake location
    center = ee.Geometry.Point([-121.900, 42.400])
    roi = center.buffer(250).bounds()  # 500m x 500m box
    
    # Test different date ranges
    date_ranges = [
        ('2015-07-01', '2016-01-01'),  # First 6 months
        ('2016-01-01', '2017-01-01'),  # 2016
        ('2017-01-01', '2018-01-01'),  # 2017
        ('2018-01-01', '2019-01-01'),  # 2018
        ('2015-07-01', '2025-12-31')   # Full range
    ]
    
    # Test different Sentinel-2 collections
    collections = [
        ('COPERNICUS/S2_SR_HARMONIZED', 'S2 SR Harmonized'),
        ('COPERNICUS/S2_SR', 'S2 SR (Legacy)'),
        ('COPERNICUS/S2', 'S2 TOA')
    ]
    
    for col_id, col_name in collections:
        print(f"\n{col_name} ({col_id}):")
        print("-" * 50)
        
        try:
            for start_date, end_date in date_ranges:
                # Basic collection without any masking
                basic_col = (ee.ImageCollection(col_id)
                           .filterDate(start_date, end_date)
                           .filterBounds(roi))
                
                count = basic_col.size().getInfo()
                print(f"  {start_date} to {end_date}: {count} scenes")
                
                if count > 0:
                    # Get first and last dates
                    sorted_col = basic_col.sort('system:time_start')
                    first_date = ee.Date(sorted_col.first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    last_date = ee.Date(sorted_col.sort('system:time_start', False).first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
                    print(f"    First scene: {first_date}")
                    print(f"    Last scene: {last_date}")
        
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # Test the impact of SCL water masking
    print(f"\n\nTesting impact of SCL water masking:")
    print("="*50)
    
    s2_col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    
    # Test without any masking
    basic_scenes = (s2_col
                   .filterDate('2015-07-01', '2019-01-01')
                   .filterBounds(roi))
    
    print(f"Total scenes (no masking): {basic_scenes.size().getInfo()}")
    
    # Test with SCL water masking
    def has_water_pixels(img):
        """Check if image has any water pixels after SCL masking"""
        scl = img.select('SCL')
        water = scl.eq(6)  # Water class
        water_eroded = water.focal_min(1)  # Erode 1 pixel
        
        # Check if any water pixels remain
        water_count = water_eroded.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=20,
            maxPixels=1e9
        ).get('SCL')
        
        return img.set('water_pixels', water_count)
    
    scenes_with_water_check = basic_scenes.map(has_water_pixels)
    
    # Filter for scenes with water pixels
    scenes_with_water = scenes_with_water_check.filter(ee.Filter.gt('water_pixels', 0))
    
    print(f"Scenes with water pixels after SCL masking: {scenes_with_water.size().getInfo()}")
    
    if scenes_with_water.size().getInfo() > 0:
        first_water_scene = ee.Date(scenes_with_water.sort('system:time_start').first().get('system:time_start')).format('YYYY-MM-dd').getInfo()
        print(f"First scene with water pixels: {first_water_scene}")
    
    # Test by year to see when water masking starts working
    print(f"\nYearly breakdown with SCL water masking:")
    print("-" * 40)
    
    for year in range(2015, 2020):
        start_date = f"{year}-01-01"
        end_date = f"{year+1}-01-01"
        
        year_scenes = (s2_col
                      .filterDate(start_date, end_date)
                      .filterBounds(roi))
        
        year_scenes_with_water = (year_scenes
                                 .map(has_water_pixels)
                                 .filter(ee.Filter.gt('water_pixels', 0)))
        
        total_count = year_scenes.size().getInfo()
        water_count = year_scenes_with_water.size().getInfo()
        
        print(f"  {year}: {total_count} total scenes, {water_count} with water pixels")

if __name__ == "__main__":
    debug_sentinel_availability()