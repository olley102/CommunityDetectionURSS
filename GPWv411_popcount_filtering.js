// JavaScript written in Google Earth Engine.

// Load GPWv411 Population Count dataset.
var dataset = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count");

// Specify start and end of timeframe.
var timeStart = ee.Date('2000-01-01');
var timeEnd = ee.Date('2025-01-01');

// Specify time interval and unit of time, e.g. weeks/months/years
var timeInterval = 5;
var timeMode = 'years';

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

// Bound each image by rectangle.
filtered = filtered.map(function(image) {
  return image.clip(geometry)
});

// Current time-step used in refreshing Map UI and visualisation shown.
var timeStep = 0;

// Display options for raster.
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

// Create text on bottom of legend.
var panel = ui.Panel({
  widgets: [
    ui.Label(raster_vis['min'])
  ],
});

legend.add(panel);

// Function to call to refresh display.
function loadTimeStep() {
  Map.clear();
  Map.addLayer(geometry, {'color': 'white'})

  // Dates of first time-step.
  var timeStepStart = timeStart.advance(timeStep * timeInterval, timeMode);
  var timeStepEnd = timeStepStart.advance(timeInterval, timeMode);

  print(timeStepStart);
  print(timeStepEnd);

  // Get image of time-step.
  var image = filtered
    .filterDate(timeStepStart, timeStepEnd)
    .first();

  // Time-step info panel.
  var timePanel = ui.Panel({
    style: {
      position: 'top-right',
      padding: '8px 15px'
    }
  });

  var timeTitle = ui.Label({
    value: 'Time-step',
    style: {
      fontWeight: 'bold',
      fontSize: '18px',
      margin: '0 0 4px 0',
      padding: '0'
    }
  });

  var timeLabel = ui.Panel({
    widgets: [
      ui.Label('Current step: ' + timeStep),
      ui.Label('Start: ' + timeStepStart.format('YYYY-MM-dd').getInfo()),
      ui.Label('End: ' + timeStepEnd.format('YYYY-MM-dd').getInfo())
    ],
  });

  timePanel.add(timeTitle);
  timePanel.add(timeLabel);

  // Button to download TIFF.
  // Export high-res TIFF that can be loaded with rasterio in python.
  var downloadBtn = ui.Button({
    label: 'Download time-step',
    style: {
      padding: '0'
    },
    onClick: function() {
      var projection = image.select('population_count').projection().getInfo();
      print('[Task] Export time-step ' + timeStep + ' to Drive.')
      Export.image.toDrive({
        image: image,
        description: 'GPW_v411_pc_time' + timeStep,
        folder: 'GoogleEarthEngine',
        crs: projection.crs,
        crsTransform: projection.transform,
        region: geometry
      });
    }
  });

  timePanel.add(downloadBtn);

  // Display first time-step.
  var raster = image.select('population_count');
  Map.addLayer(raster, raster_vis, 'population_count');
  Map.add(legend);

  // Button to view next time-step.
  var nextBtn = ui.Button({
    label: 'Next time-step',
    style: {
      padding: '0'
    },
    onClick: function() {
      timeStep += 1;
      loadTimeStep();
    }
  });
  timePanel.add(nextBtn);

  var prevBtn = ui.Button({
    label: 'Previous time-step',
    style: {
      padding: '0'
    },
    onClick: function() {
      timeStep -= 1;
      loadTimeStep();
    }
  });
  timePanel.add(prevBtn);

  Map.add(timePanel);
}

loadTimeStep();
