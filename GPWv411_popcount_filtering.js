'''JavaScript written in Google Earth Engine.'''

// Load GPWv411 Population Count dataset.
var dataset = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count");

// Specify start and end of timeframe.
var timeStart = ee.Date('2000-01-01');
var timeEnd = ee.Date('2020-01-01');

// Specify time interval and unit of time, e.g. weeks/months/years
var timeInterval = 6;
var timeMode = 'months';

// Filter dataset by time, not interval yet.
var filtered = dataset.filterDate(timeStart, timeEnd);

// Specify spatial bounds by latitude and longitude (underestimates area).
var longStart = 97.0;
var latStart = 20.6;
var longEnd = 106.2;
var latEnd = 5.7;
var longMean = (longStart + longEnd) / 2;
var latMean = (latStart + latEnd) / 2;
var geometry = ee.Geometry.Rectangle(longStart, latStart, longEnd, latEnd);
Map.addLayer(geometry, {'color': 'white'})

// Bound each image by rectangle.
filtered = filtered.map(function(image) {
  return image.clip(geometry)
});

var image = filtered
  .filterDate(timeStart, timeStart.advance(timeInterval, timeMode))
  .first();

// Try 3... trying to figure out how to extract data from rectangle.

// var imageArray = image.sampleRectangle({
//   region: geometry,
//   defaultValue: 0
// });  // too high resolution. Does not work.

// print(imageArray.get('population_count').getInfo());

// Try 2... turns out this is the best option. Export very high-res TIFF.

var projection = image.select('population_count').projection().getInfo();

Export.image.toDrive({
  image: image,
  description: 'GPW_v411_pc_time1',
  folder: 'GoogleEarthEngine',
  crs: projection.crs,
  crsTransform: projection.transform,
  region: geometry
});

// Try 1... totally wrong

// var samples = image.sample({
//   region: geometry,
//   geometries: true,
//   numPixels: 1e4,
// });

// Export.table.toDrive({
//   collection: samples,
//   description: 'GPWv411_pc_time1',
//   folder: 'GoogleEarthEngine',
//   fileFormat: 'CSV',
//   selectors: ['population_count']
// });

// Display first time-step.
var raster = filtered
  .filterDate(timeStart, timeStart.advance(timeInterval, timeMode))
  .first()
  .select('population_count');

var raster_vis = {
  "max": 1000.0,
  "palette": [
    "ffffe7",
    "86a192",
    "509791",
    "307296",
    "2c4484",
    "000066"
  ],
  "min": 0.0
};
Map.setCenter(longMean, latMean, 5);
Map.addLayer(raster, raster_vis, 'population_count');

// Create colorbar...

// Set position of panel.
var legend = ui.Panel({
style: {
position: 'bottom-left',
padding: '8px 15px'
}
});

// Create legend title.
var legendTitle = ui.Label({
value: 'Count',
style: {
fontWeight: 'bold',
fontSize: '18px',
margin: '0 0 4px 0',
padding: '0'
}
});

// Add the title to the panel.
legend.add(legendTitle);

// Create the legend image.
var lon = ee.Image.pixelLonLat().select('latitude');
var gradient = lon.multiply((raster_vis.max-raster_vis.min)/100.0).add(raster_vis.min);
var legendImage = gradient.visualize(raster_vis);

// Create text on top of legend.
var panel = ui.Panel({
widgets: [
ui.Label(raster_vis['max'])
],
});

legend.add(panel);

// Create thumbnail from the image.
var thumbnail = ui.Thumbnail({
image: legendImage,
params: {bbox:'0,0,10,100', dimensions:'10x200'},
style: {padding: '1px', position: 'bottom-center'}
});

// Add the thumbnail to the legend.
legend.add(thumbnail);

// Create text on top of legend.
var panel = ui.Panel({
widgets: [
ui.Label(raster_vis['min'])
],
});

legend.add(panel);

Map.add(legend)
