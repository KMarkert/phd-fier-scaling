import ee
import time

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

# Define the buffer size for predictions.
# This variable determines which asset collection to load.
buffer = 50

# Load the appropriate prediction and observation Earth Engine ImageCollections.
# Predictions are loaded based on the specified buffer size.
# Observations are loaded from a fixed asset ID.
if buffer == 'baseline':
    predictions = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_baseline')
else:
    predictions = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_buffer_{buffer:02d}km')
observations = ee.ImageCollection('projects/byu-hydroinformatics-gee/assets/noaa_jpss_floods')

# Get a sorted list of unique dates from the prediction image collection.
# These dates will be used to filter and process images one by one.
dates = predictions.aggregate_array('system:time_start').sort().map(lambda d: ee.Date(d).format('YYYY-MM-dd')).getInfo()

# Iterate over each date to process and export corrected predictions.
for date in dates:
    print('Exporting:', date)
    
    # Filter the predictions for the current date and select the first image.
    # Rename the band to 'water_fraction' and retrieve the 'train' property.
    pred = predictions.filterDate(date).first().rename('water_fraction')
    train_flag = pred.get('train')
    # Clamp water fraction values between 0 and 100.
    pred = pred.clamp(0,100)

    # Load the percentile images for observations and predictions.
    # These are used for quantile matching.
    obs_ps = ee.Image('projects/byu-hydroinformatics-gee/assets/kmarkert/fier/viirs_water_fraction_obs_percentiles')
    if buffer == 'baseline':
        pred_ps = ee.Image(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_pred_baseline_percentiles_v2')
    else:
        pred_ps = ee.Image(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_pred_buffer_{buffer:02d}km_percentiles_v2')

    # Calculate the percentile index for the predicted water fraction.
    # This finds the index of the closest percentile in `pred_ps` for each pixel in `pred`.
    pred_ps_arr = pred_ps.toArray()
    idx = pred_ps_arr.subtract(pred).abs().multiply(-1).arrayArgmax().arrayFlatten([['idx']])

    # Fill the percentile index with a focal mode to smooth out isolated pixels.
    idx_buff = idx.focal_mode(radius= 4.5, iterations= 5)

    # Unmask the original percentile index with the filled values,
    # effectively filling in masked areas with smoothed values.
    idx_filled = idx.unmask(idx_buff)

    # Create an ImageCollection of matched water fraction values using the original percentile index.
    # For each percentile (0-100), select the corresponding band from `obs_ps` and mask it
    # where `idx` matches the current percentile. Then mosaic all these images.
    matched = ee.ImageCollection.fromImages(
        ee.List.sequence(0, 100).map(
            lambda i: obs_ps.select(ee.Number(i).int()).updateMask(idx.eq(ee.Number(i).int())).rename('water_fraction_matched')
        )
    ).mosaic()

    # Create an ImageCollection of matched water fraction values using the filled percentile index.
    # Similar to `matched`, but uses `idx_filled` for smoother results.
    matched_filled = ee.ImageCollection.fromImages(
        ee.List.sequence(0, 100).map(
            lambda i: obs_ps.select(ee.Number(i).int()).updateMask(idx_filled.eq(ee.Number(i).int())).rename('water_fraction_matched_filled')
        )
    ).mosaic()

    # Define the output Earth Engine asset collection path based on the buffer size.
    if buffer == 'baseline':
        outcollection = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_baseline_corrected_v2'
    else:
        outcollection = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_buffer_{buffer:02d}km_corrected_v2'

    # Attempt to create the asset collection. If it already exists, catch the exception and continue.
    try: 
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, outcollection)
    except ee.EEException:
        # Pass if the asset already exists, otherwise re-raise the exception.
        pass

    # Export the combined matched and matched_filled images to an Earth Engine asset.
    # Set system properties like time_start and train flag.
    export = ee.batch.Export.image.toAsset(
        image = matched.addBands(matched_filled).set('system:time_start', ee.Date(date).millis(), 'train', train_flag),
        assetId=f'{outcollection}/fier_test_corrected_{date.replace("-", "")}',
        description=f'fier_test_corrected_{date.replace("-", "")}',
        region = pred.geometry().bounds(), # Export region is the bounds of the prediction image.
        crs = 'EPSG:4326', # Coordinate Reference System.
        scale=pred.projection().nominalScale().getInfo(), # Nominal scale of the prediction image.
        maxPixels=1e13, # Maximum number of pixels to export.
    )
    export.start() # Start the export task.

    # Pause for 1 second to avoid overwhelming the Earth Engine API.
    time.sleep(1)
