import ee

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

def to_water_fraction(img):
  """Converts the VIIRS water fraction dataset to actual water fraction units [0-100].

  This function processes an Earth Engine image representing VIIRS water detection
  and converts its pixel values into a continuous water fraction percentage (0-100).
  It handles permanent water (99) and other valid water detection codes (16, 17).

  Args:
    img: An ee.Image representing VIIRS water detection.

  Returns:
    An ee.Image representing water fraction, with values clamped between 0 and 100,
    and masked to valid observations.
  """
  # Create a mask for valid observations (values 99, 16, or 17, or >= 100).
  valid_mask = img.gte(99).Or(img.eq(16).Or(img.eq(17)))
  
  # Identify permanent water (value 99) and set its fraction to 100.
  water_permanent = img.eq(99).multiply(100)
  
  # Calculate water fraction for values >= 100 by subtracting 100.
  # Unmask with 0 to ensure non-water areas are 0.
  water_fraction = img.updateMask(img.gte(100)).subtract(100).unmask(0)

  # Combine permanent water and calculated water fraction, apply valid mask,
  # convert to float, and set the system:time_start property.
  return water_permanent.add(water_fraction) \
    .updateMask(valid_mask) \
    .float() \
    .set({'system:time_start': img.date().millis()})


def calculate_ssim(reference, matched):
  """Calculates the Structural Similarity Index (SSIM) between two images.

  SSIM is a perceptual metric that quantifies the similarity between two images.
  It is based on three comparison measurements: luminance, contrast, and structure.

  Args:
    reference: An ee.Image representing the reference (ground truth) image.
    matched: An ee.Image representing the image to be compared against the reference.

  Returns:
    An ee.Image representing the SSIM between the two images, with values typically
    ranging from -1 to 1, where 1 indicates perfect similarity.
  """

  # Define constants and a Gaussian kernel for SSIM calculation.
  kernel = ee.Kernel.gaussian(5.5) # Gaussian kernel with a standard deviation of 5.5 pixels.
  k1 = ee.Image(0.01) # Constant for luminance stability.
  k2 = ee.Image(0.03) # Constant for contrast stability.
  L = ee.Image(100).subtract(1) # Dynamic range of pixel values (assuming 0-100).

  # Calculate constants for SSIM formula.
  c1 = k1.multiply(L).pow(2)
  c2 = k2.multiply(L).pow(2)

  # Calculate mean and variance for reference (mux, sigmax) and matched (muy, sigmay) images
  # within the defined kernel neighborhood.
  mux = reference.reduceNeighborhood(
    reducer= ee.Reducer.mean(),
    kernel= kernel
  )
  muy = matched.reduceNeighborhood(
    reducer= ee.Reducer.mean(),
    kernel= kernel
  )
  sigmax = reference.reduceNeighborhood(
    reducer= ee.Reducer.variance(),
    kernel= kernel
  )
  sigmay = matched.reduceNeighborhood(
    reducer= ee.Reducer.variance(),
    kernel= kernel
  )

  # Calculate covariance between the reference and matched images.
  # This measures how much two variables change together.
  covar = reference.subtract(mux).addBands(matched.subtract(muy)).toArray().reduceNeighborhood(
    reducer= ee.Reducer.centeredCovariance(),
    kernel= kernel
  ).arrayFlatten([['inner','cross'],['cross','inner']]).select('cross_cross')

  # Calculate SSIM components based on the formula.
  a = ee.Image(2).multiply(mux).multiply(muy).add(c1)
  b = ee.Image(2).multiply(covar).add(c2)
  c = mux.pow(2).add(muy.pow(2)).add(c1)
  d = sigmax.add(sigmay).add(c2)

  # Calculate SSIM.
  ssim = a.multiply(b).divide(c.multiply(d))

  return ssim

def calc_err_map(image):
  """Calculates various error metrics between a predicted image and observed water fraction.

  This function takes a predicted water fraction image, retrieves the corresponding
  observation, and computes absolute error, squared error, squared observation,
  RMSE, RRMSE, and SSIM.

  Args:
    image: An ee.Image representing the predicted water fraction.

  Returns:
    An ee.Image containing bands for 'absolute_error', 'sq_error', 'sq_obs', and 'ssim'.
  """
  date = image.date()
  # Filter observations for the corresponding date and select the 'water_detection' band.
  reference = observations.filterDate(date).first().select('water_detection')
  
  # Calculate squared error: (prediction - reference)^2.
  sq_error = image.subtract(reference).pow(2).rename('sq_error')

  # Calculate absolute error: |prediction - reference|.
  absolute_error = (image.subtract(reference)).abs().rename('absolute_error')

  # Calculate squared observation: reference^2.
  sq_obs = reference.pow(2).rename('sq_obs')

  # Calculate Root Mean Squared Error (RMSE).
  rmse = sq_error.sqrt().rename('rmse')

  # Calculate Relative Root Mean Squared Error (RRMSE).
  rrsme = (rmse.divide(sq_obs)).sqrt().multiply(100).rename('rrmse')

  # Calculate Structural Similarity Index (SSIM).
  ssim = calculate_ssim(reference, image).rename('ssim')

  # Concatenate all calculated error bands into a single image.
  return ee.Image.cat([absolute_error, sq_error, sq_obs, ssim])

# Set the buffer size for predictions.
buffer = 50
# Flag to indicate if corrected predictions are used.
corrected = True
# Flag to indicate if the Mississippi region is used for export.
ms_region = False

print(f'Buffer: {buffer},\nCorrected: {corrected},\nMS Region: {ms_region}')

# Determine the suffix for asset IDs based on whether corrected predictions are used.
if corrected:
  suffix = '_corrected'
else:
  suffix = ''

# Load the necessary prediction and observation Earth Engine ImageCollections.
# Predictions are loaded based on the specified buffer size and correction status.
# Observations are loaded from a fixed asset ID.
if buffer == 'baseline':
  predictions = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_baseline{suffix}')
else:
  predictions = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_buffer_{buffer:02d}km{suffix}')
  
observations = ee.ImageCollection('projects/byu-hydroinformatics-gee/assets/noaa_jpss_floods')

# Apply the `to_water_fraction` function to the observations image collection.
observations = observations.map(to_water_fraction)

# Load the Mississippi AOI (Area of Interest) feature.
ms_region_feature = ee.Feature(
    ee.FeatureCollection('projects/byu-hydroinformatics-gee/assets/kmarkert/fier/mississippi_aoi')
    .first()
)

# Determine the export region. If `ms_region` is True, use the Mississippi AOI,
# otherwise use the geometry of the first prediction image.
if ms_region:
  export_region = ms_region_feature.geometry()
else:
  export_region = predictions.first().geometry()

# Select the first band of predictions and filter for non-training data.
predictions = predictions.select(0).filter(ee.Filter.eq('train', 0))

# Calculate the error map for each prediction image and reduce them by taking the mean.
# The '16' in reduce(ee.Reducer.mean(), 16) refers to the number of input bands to reduce.
err_img = predictions.map(calc_err_map).reduce(ee.Reducer.mean(),16)

# Calculate overall RMSE from the mean squared error.
rmse = err_img.select('sq_error_mean').sqrt().rename('rmse')

# Calculate overall RRMSE from the mean squared error and mean squared observation.
rrmse = err_img.select('sq_error_mean').divide(err_img.select('sq_obs_mean')).sqrt().rename('rrmse')

# Concatenate the selected error metrics into a single image for export.
export_img = ee.Image.cat([
  err_img.select(['ssim_mean','absolute_error_mean'],['ssim','absolute_error']),
  rmse,
  rrmse
])

# Define the output asset name based on buffer size and correction status.
if buffer == 'baseline':
    outname = f'fier_err_stats_baseline{suffix}'
else:
    outname = f'fier_err_stats_buffer_{buffer:02d}km{suffix}'
  
# Export the error map image to an Earth Engine asset.
task = ee.batch.Export.image.toAsset(
    image= export_img,
    description= outname,
    assetId= f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/errors/{outname}',
    region= export_region,
    scale= observations.first().projection().nominalScale().getInfo(),
    maxPixels= 1e13
)
task.start() # Start the export task.
