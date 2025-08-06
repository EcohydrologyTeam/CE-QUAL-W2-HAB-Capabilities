/**************************************************************************
 *                     Detroit Lake, Oregon
/*                        Sentinel Chl-a
/**************************************************************************/

// Lake center point and output filename
var sample_coordinates = [-122.184, 44.711]
var outfile = 'Detroit_NDCI_Sentinel2_2011-2025'

// Lake center point
var sample_location = ee.Geometry.Point(sample_coordinates);      // mid-lake pixel
Map.centerObject(sample_location, 11);
Map.addLayer(sample_location, {color: 'yellow'}, 'Sample Point');

var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterDate('2011-01-01', ee.Date(Date.now()))
  .filterBounds(sample_location)
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100))
  .map(function(img){
    var ndci = img.normalizedDifference(['B5','B4'])  // RE2 – Red
                 .rename('ndci');
    return ndci.set('system:time_start', img.get('system:time_start'));
  });

print('Sentinel-2 scene count:', s2.size());

var s2Series = s2.map(function(img){
  var mean = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: sample_location,
      scale: 10,
      maxPixels: 1e9});
  return ee.Feature(null, {'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
                           'ndci': mean.get('ndci')});
});
Export.table.toDrive({
  collection: s2Series,
  description: outfile,
  fileFormat: 'CSV'
});