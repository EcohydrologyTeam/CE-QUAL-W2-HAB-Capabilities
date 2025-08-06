/**************************************************************************
 *                     Upper Klamath Lake, Oregon
/*                        Landsat Chl-a
/**************************************************************************/

// Lake center point and output filename
var sample_coordinates = [-122, 42.455]
var outfile = 'Klamath_NDCI_Landsat_2011-2025'

// Lake center point
var sample_location = ee.Geometry.Point(sample_coordinates); // mid-lake pixel
Map.centerObject(sample_location, 11);
Map.addLayer(sample_location, {color: 'yellow'}, 'Sample Point');

// Calibration parameters
var slope = 0.0000275;
var offset = -0.2;

// Landsat-8 SR collection: starts 2013-04-11
var l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterDate('2011-01-01', ee.Date(Date.now()))
  .filterBounds(sample_location)
  .map(function(img){
    // Convert DN to surface reflectance
    var refl = img.multiply(slope).add(offset);
    var ndci = refl.normalizedDifference(['SR_B5','SR_B4'])  // NIR – Red
                 .rename('ndci');
    return ndci.set('system:time_start', img.get('system:time_start'));
  });

print('Landsat scene count:', l8.size());

// Lake-mean time-series
var l8Series = l8.map(function(img){
  var mean = img.reduceRegion({
      reducer: ee.Reducer.mean(),
      geometry: sample_location,
      scale: 30,
      maxPixels: 1e9});
  return ee.Feature(null, {'date': ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
                           'ndci': mean.get('ndci')});
});
Export.table.toDrive({
  collection: l8Series,
  description: outfile,
  fileFormat: 'CSV'
});

