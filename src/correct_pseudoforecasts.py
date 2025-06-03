import ee

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

# Define the buffer size for pseudoforecasts.
buffer = 1

# Load the pseudoforecast predictions.
# Filters for images where 'corrected' property is 0 (not yet corrected).
predictions = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_pseudoforecasts_buffer_{buffer:02d}km').filter('corrected==0')

# Load the observation image collection.
observations = ee.ImageCollection('projects/byu-hydroinformatics-gee/assets/noaa_jpss_floods')

# Get a list of system:index (image IDs) from the predictions collection.
img_ids = predictions.aggregate_array('system:index').getInfo()

# Iterate over each image ID to process and export corrected pseudoforecasts.
for img_id in img_ids:
    print('Exporting:', img_id)

    # Filter the predictions for the current image ID and select the first image.
    # Rename the band to 'water_fraction'.
    pred_ = predictions.filter(ee.Filter.eq('system:index', img_id)).first().rename('water_fraction')
    
    # Retrieve forecast and date properties from the prediction image.
    FORECAST = pred_.get('forecast').getInfo()
    date = pred_.date().format('YYYY-MM-dd').getInfo()

    # Clamp water fraction values between 0 and 100.
    pred = pred_.clamp(0,100)

    # Load the percentile images for observations and predictions.
    # These are used for quantile matching.
    obs_ps = ee.Image('projects/byu-hydroinformatics-gee/assets/kmarkert/fier/viirs_water_fraction_obs_percentiles')
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

    # Define the output Earth Engine asset collection path.
    outcollection = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_pseudoforecasts_buffer_{buffer:02d}km'

    # Combine the matched and matched_filled bands, clamp values, and copy properties from the original image.
    # Set the 'corrected' property to 1 to indicate that this image has been corrected.
    out_img = ee.Image(
        matched.addBands(matched_filled)
        .clamp(0,100)
        .copyProperties(pred_, ['system:time_start', 'system:time_end', 'forecast', 'forecast_offset', 'reference_date', 'corrected', 'buffer_size'])
    ).set('corrected', 1)

    # Attempt to create the asset collection. If it already exists, catch the exception and continue.
    try: 
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, outcollection)
    except ee.EEException:
        # Pass if the asset already exists, otherwise re-raise the exception.
        pass

    # Export the corrected image to an Earth Engine asset.
    export = ee.batch.Export.image.toAsset(
        image = out_img,
        assetId=f'{outcollection}/fier_{date.replace("-", "")}_{FORECAST}_corrected',
        description=f'fier{date.replace("-", "")}_{FORECAST}_corrected',
        region = pred_.geometry().bounds(), # Export region is the bounds of the prediction image.
        crs = 'EPSG:4326', # Coordinate Reference System.
        scale=pred_.projection().nominalScale().getInfo(), # Nominal scale of the prediction image.
        maxPixels=1e13, # Maximum number of pixels to export.
    )
    export.start() # Start the export task.
