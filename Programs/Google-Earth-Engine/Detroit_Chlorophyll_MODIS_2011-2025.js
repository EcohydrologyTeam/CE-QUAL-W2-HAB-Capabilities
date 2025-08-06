/**************************************************************************
 *                     Detroit Lake, Oregon
/*                       MODIS 500 m Chl-a
/**************************************************************************/

// Lake center point and output filename
var sample_coordinates = [-122.184, 44.711]
var terra_outfile = 'Detroit_MODIS_Terra_500m_Chl_singlePixel'
var aqua_outfile = 'Detroit_MODIS_Aqua_500m_Chl_singlePixel'

// Lake center point
var sample_location = ee.Geometry.Point(sample_coordinates);      // mid-lake pixel
Map.centerObject(sample_location, 11);
Map.addLayer(sample_location, {color: 'yellow'}, 'Sample Point');

// Green / Red chlorophyll algorithm (units: µg L-¹)
function addGR_Chl(img) {
  // MODIS SR 500 m bands (scale factor 0.0001)
  var sr555 = img.select('sur_refl_b04').multiply(0.0001);   // 555 nm
  var sr645 = img.select('sur_refl_b01').multiply(0.0001);   // 645 nm

  // Approximate remote-sensing reflectance: R_rs ≈ SR / π
  var rrs555 = sr555.divide(Math.PI);
  var rrs645 = sr645.divide(Math.PI);

  var offset = 1.54; // Global offset
//  var offset = 0.96; // sets ~5 µg L-¹ when ratio ≈ 0.7
  var slope = 1.70; // Global slope
  
  var logRatio = rrs555.divide(rrs645).log10();               // log10(555 / 645)
  var log10Chl = ee.Image(offset).add(logRatio.multiply(slope)); // calibration
  var chl      = ee.Image(10).pow(log10Chl).rename('chlor_a'); // µg L-¹

  return img.addBands(chl);
}

function maskLightCloud(img){
  var qa = img.select('state_1km');          // Global Average product has this band
  var cloud = qa.bitwiseAnd(1 << 10).neq(0);  // bit 10 = cirrus / cloud
  var snow  = qa.bitwiseAnd(1 << 12).neq(0);  // bit 12 = snow/ice
  return img.updateMask(cloud.not()).updateMask(snow.not());
}

// Daily MODIS SR collections (2011-01-01 to today)
var terra = ee.ImageCollection('MODIS/006/MOD09GA')
    .filterDate('2011-01-01', ee.Date(Date.now()))
    .map(maskLightCloud)
    .map(addGR_Chl)
    .select('chlor_a');

var aqua  = ee.ImageCollection('MODIS/006/MYD09GA')
    .filterDate('2011-01-01', ee.Date(Date.now()))
    .map(maskLightCloud)
    .map(addGR_Chl)
    .select('chlor_a');

// Sample a single pixel
function imgToFeature(img, sensor) {
  var fc = img.sample({
      region     : sample_location,
      scale      : 500,
      numPixels  : 1,
      geometries : false
  });
  return ee.FeatureCollection(ee.Algorithms.If(
      fc.size().gt(0),
      fc.map(function(f){
        return ee.Feature(null, {
          date   : ee.Date(img.get('system:time_start')).format('YYYY-MM-dd'),
          chl    : f.get('chlor_a'),
          sensor : sensor
        });
      }),
      ee.FeatureCollection([])   // Image masked at point. Return empty collection
  ));
}

// Build the two time-series (flat collections, no nulls)
var terraSeries = terra.map(function(i){ return imgToFeature(i, 'Terra'); }).flatten();
var aquaSeries  = aqua .map(function(i){ return imgToFeature(i, 'Aqua');  }).flatten();

/// Quick counts
print('Terra samples:', terraSeries.size());
print('Aqua  samples:', aquaSeries.size());

// Plot charts
print(ui.Chart.feature.byFeature(aquaSeries,  'date', 'chl')
        .setChartType('ScatterChart')
        .setOptions({title:'Aqua MODIS 500 m Chl-a (single pixel)'}));
print(ui.Chart.feature.byFeature(terraSeries, 'date', 'chl')
        .setChartType('ScatterChart')
        .setOptions({title:'Terra MODIS 500 m Chl-a (single pixel)'}));

// Export to Google Drive
Export.table.toDrive({
  collection : terraSeries,
  description: terra_outfile,
  fileFormat : 'CSV'
});

Export.table.toDrive({
  collection : aquaSeries,
  description: aqua_outfile,
  fileFormat : 'CSV'
});
