// ==========================================
// 1. 全局配置与物候期设定
// ==========================================

var years = [2016, 2017, 2018, 2019, 2020, 2021]; 
var targetZones = [1, 2, 3, 4, 5]; 

var phenologyDict = {
  1: { 'P1': '09/15-10/05', 'P2': '11/05-03/15', 'P3': '03/20-04/25', 'P4': '05/05-05/25', 'P5': '06/05-06/25' },
  2: { 'P1': '10/25-11/25', 'P2': '11/26-02/04', 'P3': '02/05-03/15', 'P4': '03/20-04/05', 'P5': '04/15-05/05' },
  3: { 'P1': '10/25-11/20', 'P2': '12/15-02/05', 'P3': '02/15-03/25', 'P4': '04/05-04/25', 'P5': '05/01-05/25' },
  4: { 'P1': '10/05-11/15', 'P2': '12/15-02/15', 'P3': '02/25-04/15', 'P4': '04/20-05/05', 'P5': '05/10-06/05' },
  5: { 'P1': '09/25-10/15', 'P2': '11/15-03/10', 'P3': '03/15-04/25', 'P4': '05/05-05/15', 'P5': '05/20-06/15' }
};

// 导出列定义（已移除 NIRv_AUC）
var staticBands = ['Elevation', 'Slope', 'Aspect', 'Sand', 'Clay', 'BD', 'pH', 'SOC'];
var dynamicBands = ['EVI', 'EVI_max', 'FDD', 'GDD', 'NDVI', 'NDVI_max', 'NDWI', 'NIRv', 'PPT', 'SM', 'Tmean', 'VPD', 'VPD_max'];
var expectedPeriods = ['P1', 'P2', 'P3', 'P4', 'P5'];
var exportSelectors = ['Year', 'Zone', 'latitude', 'longitude', 'yield'].concat(staticBands);
expectedPeriods.forEach(function(p) {
  dynamicBands.forEach(function(db) { exportSelectors.push(p + '_' + db); });
});

// ==========================================
// 2. 核心预处理函数 (Academic Standards)
// ==========================================

// [静态底图] ISRIC SoilGrids v2 + SRTM + Resampling
var getStaticFeatures = function() {
  var dem = ee.Image("USGS/SRTMGL1_003");
  var elevation = dem.select('elevation').rename('Elevation');
  var slope = ee.Terrain.slope(dem).rename('Slope');
  var aspect = ee.Terrain.aspect(dem).rename('Aspect');
  
  var soilBase = "projects/soilgrids-isric/";
  var getSoil = function(name, newName, scale) {
    return ee.Image(soilBase + name + "_mean").select(name + "_0-5cm_mean")
             .multiply(scale).resample('bilinear').rename(newName);
  };

  return ee.Image.cat([
    elevation, slope, aspect,
    getSoil('sand', 'Sand', 0.1), getSoil('clay', 'Clay', 0.1),
    getSoil('bdod', 'BD', 0.01), getSoil('phh2o', 'pH', 0.1),
    getSoil('soc', 'SOC', 0.1)
  ]).toFloat();
};

// [气象处理] ERA5-Land + 双线性插值
var getMeteorologicalFeatures = function(startDate, endDate) {
  var era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    .filterDate(startDate, endDate)
    .select(['temperature_2m', 'dewpoint_temperature_2m', 'total_precipitation_sum', 'volumetric_soil_water_layer_1']);
  
  var daily = era5.map(function(img) {
    var tmean = img.select('temperature_2m').subtract(273.15);
    var tdew = img.select('dewpoint_temperature_2m').subtract(273.15);
    var e_s = tmean.multiply(17.27).divide(tmean.add(237.3)).exp().multiply(0.6108);
    var e_a = tdew.multiply(17.27).divide(tdew.add(237.3)).exp().multiply(0.6108);
    var vpd = e_s.subtract(e_a).max(0).rename('VPD');
    return img.addBands([
      tmean.rename('Tmean'), 
      img.select('total_precipitation_sum').multiply(1000).rename('PPT'),
      img.select('volumetric_soil_water_layer_1').rename('SM'),
      tmean.where(tmean.lt(0), 0).rename('GDD'),
      ee.Image(0).subtract(tmean).where(tmean.gte(0), 0).rename('FDD'),
      vpd
    ]);
  });

  var sumImg = daily.select(['GDD', 'FDD', 'PPT']).sum();
  var meanImg = daily.select(['SM', 'VPD', 'Tmean']).mean();
  var maxImg = daily.select(['VPD']).max().rename('VPD_max');
  
  return sumImg.addBands([meanImg, maxImg]).resample('bilinear').toFloat();
};

// [遥感处理] 去云 + SBAF 校准 + 中位数合成
var getRSFeatures = function(startDate, endDate, year, zonePoints) {
  // 1. MODIS 填补源 (双线性插值)
  var modis = ee.ImageCollection("MODIS/061/MOD13Q1").filterDate(startDate, endDate)
    .map(function(img) {
      var ndvi = img.select('NDVI').multiply(0.0001);
      var evi = img.select('EVI').multiply(0.0001);
      var nir = img.select('sur_refl_b02').multiply(0.0001);
      var swir = img.select('sur_refl_b07').multiply(0.0001);
      return img.select().addBands([
        ndvi.rename('NDVI'), evi.rename('EVI'),
        nir.subtract(swir).divide(nir.add(swir)).rename('NDWI'),
        ndvi.multiply(nir).rename('NIRv')
      ]);
    });
  var fallbackMedian = modis.median().resample('bilinear');
  var fallbackMax = modis.select(['NDVI', 'EVI']).max().resample('bilinear').rename(['NDVI_max', 'EVI_max']);

  // 2. 高分主干源
  var backbone;
  if (year <= 2018) {
    backbone = ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
      .filterBounds(zonePoints).filterDate(startDate, endDate)
      .map(function(img) {
        var qa = img.select('QA_PIXEL');
        var mask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
        var scaled = img.select(['SR_B4', 'SR_B5', 'SR_B6']).multiply(0.0000275).add(-0.2);
        // SBAF 校准 (OLI to MSI)
        var red = scaled.select('SR_B4').multiply(0.9778).add(0.0068);
        var nir = scaled.select('SR_B5').multiply(1.0053).add(-0.0009);
        var swir = scaled.select('SR_B6').multiply(0.9755).add(0.0045);
        var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        return img.select().addBands([ndvi, nir.subtract(swir).divide(nir.add(swir)).rename('NDWI'), ndvi.multiply(nir).rename('NIRv')]).updateMask(mask);
      });
  } else {
    backbone = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
      .filterBounds(zonePoints).filterDate(startDate, endDate)
      .map(function(img) {
        var scl = img.select('SCL');
        var mask = scl.gte(4).and(scl.lte(7));
        var scaled = img.select(['B4', 'B8', 'B11']).divide(10000);
        var red = scaled.select('B4'), nir = scaled.select('B8'), swir = scaled.select('B11');
        var ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI');
        return img.select().addBands([ndvi, nir.subtract(swir).divide(nir.add(swir)).rename('NDWI'), ndvi.multiply(nir).rename('NIRv')]).updateMask(mask);
      });
  }

  var bMedian = backbone.median();
  var bMax = backbone.select('NDVI').max().rename('NDVI_max');

  return ee.Image.cat([
    bMedian.select('NDVI').unmask(fallbackMedian.select('NDVI')),
    bMax.unmask(fallbackMax.select('NDVI_max')),
    bMedian.select('NDWI').unmask(fallbackMedian.select('NDWI')),
    bMedian.select('NIRv').unmask(fallbackMedian.select('NIRv')),
    fallbackMedian.select('EVI'),
    fallbackMax.select('EVI_max')
  ]).toFloat();
};

// ==========================================
// 3. 任务执行循环
// ==========================================

var staticImage = getStaticFeatures();

years.forEach(function(year) {
  targetZones.forEach(function(zoneId) {
    var assetPath = 'projects/project-for-forecast/assets/YearandZone/Wheat_Points_Zone' + zoneId + '_' + year;
    var zonePoints = ee.FeatureCollection(assetPath).map(function(f) {
      var coords = f.geometry().coordinates();
      return f.set({'longitude': coords.get(0), 'latitude': coords.get(1), 'Year': year, 'Zone': zoneId});
    });

    var dates = phenologyDict[zoneId];
    if (!dates) return;

    var periodImages = [];
    Object.keys(dates).forEach(function(period) {
      var p = dates[period].split('-');
      var startM = parseInt(p[0].split('/')[0]), endM = parseInt(p[1].split('/')[0]);
      var startY = (startM >= 8) ? year - 1 : year;
      var endY = (endM >= 8 && startM < 8) ? year : (startM >= 8 && endM < 8) ? year : (startM >= 8) ? year - 1 : year;
      var sd = ee.Date.fromYMD(startY, startM, parseInt(p[0].split('/')[1]));
      var ed = ee.Date.fromYMD(endY, endM, parseInt(p[1].split('/')[1])).advance(1, 'day');

      var met = getMeteorologicalFeatures(sd, ed);
      var rs = getRSFeatures(sd, ed, year, zonePoints);
      
      var combined = ee.Image.cat([met, rs]);
      periodImages.push(combined.rename(combined.bandNames().map(function(n) { return ee.String(period).cat('_').cat(n); })));
    });

    var finalImg = ee.Image.cat([staticImage].concat(periodImages)).unmask(-9999);
    
    var extracted = finalImg.reduceRegions({
      collection: zonePoints,
      reducer: ee.Reducer.first(),
      scale: 30,
      tileScale: 16
    }).map(function(f) { return ee.Feature(null, f.toDictionary()); });

    var taskName = 'Academic_Zone' + zoneId + '_' + year;
    Export.table.toDrive({
      collection: extracted,
      description: taskName,
      folder: 'Wheat_Features_Academic_Verified', 
      fileNamePrefix: taskName,
      fileFormat: 'CSV',
      selectors: exportSelectors
    });
  });
});