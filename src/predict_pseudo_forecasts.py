from google.cloud.bigquery import Client, QueryJobConfig
import ee
import datetime
import numpy as np
from pathlib import Path
import xarray as xr
import pandas as pd

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

# Initialize Google BigQuery client with a specified project.
client = Client(project='kmarkert-personal')

# Load NWM streamflow data, rename 'streamflow' to 'hydro_var', and drop duplicate feature IDs.
q = xr.open_dataset('../nwm_streamflow/nwm_streamflow_daily.nc').rename({'streamflow':'hydro_var'}).drop_duplicates(dim='feature_id')

# Table lookup for National Water Model forecast types in BigQuery.
nwm_table_lookup = {
    'nowcast': 'analysis_assim',
    'medium_range': 'medium_range',
    'long_range': 'long_range'
}

# Forecast offset lookup in days for different NWM forecast types.
forecast_offset_lookup = {
    'nowcast': 1,
    'medium_range': 7,
    'long_range': 15
}

# Set the desired forecast type (e.g., 'nowcast', 'medium_range', 'long_range').
FORECAST = 'nowcast'

# Define specific dates for pseudoforecast generation.
dates = [
    '2019-02-25', # Example: 50-year flood event.
    '2020-02-15', # Example: 5-year flood event.
    '2019-09-29', # Example: Minimum flow option for 2019.
    '2020-10-21', # Example: Minimum flow for 2020.
]

# Select the date for the current pseudoforecast run.
DATE = dates[0]

# Calculate the reference date based on the selected date and forecast offset.
reference_date = datetime.datetime.strptime(DATE, '%Y-%m-%d') - datetime.timedelta(days=forecast_offset_lookup[FORECAST])

# Load the HUC8 ID image from Earth Engine.
huc8Id_image = ee.Image('projects/byu-hydroinformatics-gee/assets/NOAA_FIER/huc8Id_image')

# Load the FIER scaling image collection for a specific buffer size (01km in this case).
fier_collection = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/fier_scaling_buffer_01km')

# Get a list of HUC8 IDs present in the FIER collection.
hydroids = fier_collection.aggregate_array('huc8Id').getInfo()

# Print current processing status.
date_str = DATE
print('Exporting:', date_str, 'Forecast:', FORECAST)

def extract_coefs(img, mode_names):
    """
    Extracts polynomial coefficients for each hydrological mode from an Earth Engine image.

    Args:
        img (ee.Image): An Earth Engine image containing mode coefficients (e.g., from fier_scaling).
        mode_names (ee.List): A list of mode names (e.g., 'mode_0', 'mode_1').

    Returns:
        ee.Array: An Earth Engine Array containing the coefficients for each mode.
                  Each row corresponds to a mode, and columns are coefficients (coeff_0, coeff_1, coeff_2).
    """
    def to_array(mode):
        mode = ee.String(mode)
        
        # Retrieve coefficients for the current mode.
        coef1 = img.get(mode.cat('_coeff_0'))
        coef2 = img.get(mode.cat('_coeff_1'))
        coef3 = img.get(mode.cat('_coeff_2'))

        return ee.List([coef1, coef2, coef3])

    # Map the `to_array` function over the list of mode names to get coefficients for all modes.
    x = mode_names.map(to_array)

    return ee.Array(x)

def extract_reaches(img, mode_names):
    """
    Extracts associated reach IDs for each hydrological mode from an Earth Engine image.

    Args:
        img (ee.Image): An Earth Engine image containing reach IDs (e.g., from fier_scaling).
        mode_names (ee.List): A list of mode names.

    Returns:
        ee.List: An Earth Engine List containing the reach IDs for each mode.
    """
    # Map over mode names to extract the corresponding reach ID for each mode.
    x = mode_names.map(lambda x: img.get(ee.String(x).cat('_reach')))

    return x

imgs = [] # List to store reconstructed images for each HUC8 watershed.

# Iterate over each HUC8 watershed ID.
for hydroid in hydroids:
    # Filter the FIER collection for the current HUC8 ID and select the first image.
    img = fier_collection.filter(ee.Filter.eq('huc8Id', hydroid)).first()

    # Get band names and remove 'center' to get only mode names.
    mode_names = img.bandNames().remove('center')

    # Extract reach IDs associated with the modes for the current HUC8.
    reaches_ = extract_reaches(img, mode_names)
    reaches = reaches_.getInfo()

    # Handle cases where only one reach is found by duplicating it for tuple creation.
    if len(reaches) == 1:
        reaches_tuple = (reaches[0], reaches[0])
    else:
        reaches_tuple = tuple(reaches)

    # Construct BigQuery SQL query based on the forecast type.
    # This query fetches streamflow data for the specified feature IDs and date.
    nowcast_query = f"""
        SELECT
            feature_id,
            AVG(streamflow) as streamflow
        FROM
            `bigquery-public-data.national_water_model.{nwm_table_lookup[FORECAST]}_channel_rt`
        WHERE
            feature_id IN {reaches_tuple}
            AND time BETWEEN TIMESTAMP('{DATE} 00:00:00') AND TIMESTAMP('{DATE} 23:59:59')
            AND forecast_offset = 1
        GROUP BY
            feature_id
    """

    forecast_query = f"""
        SELECT
            feature_id,
            AVG(streamflow) as streamflow
        FROM
            `bigquery-public-data.national_water_model.{nwm_table_lookup[FORECAST]}_channel_rt`
        WHERE
            feature_id IN {reaches_tuple}
            AND time BETWEEN TIMESTAMP('{DATE} 00:00:00') AND TIMESTAMP('{DATE} 23:59:59')
            AND reference_time = TIMESTAMP('{reference_date.strftime('%Y-%m-%d')} 00:00:00')
        GROUP BY
            feature_id
    """

    if FORECAST == 'nowcast':
        query = nowcast_query
    else:
        query = forecast_query

    # Execute the BigQuery query and convert the result to a Pandas DataFrame.
    job = client.query(query)
    df = job.to_dataframe()

    # Set 'feature_id' as the index and drop the column.
    df.index = df['feature_id']
    df = df.drop(columns='feature_id')

    # Convert the Pandas DataFrame to an xarray Dataset.
    q_date = df.to_xarray()

    # Fill any missing streamflow values using historical data if available.
    if any(q_date.streamflow.isnull()):
        q_fill = q.hydro_var.sel(time=date_str).sel(feature_id=q_date.feature_id.values)
        q_date = q_date.fillna(q_fill)

    # Assign the current date to the 'time' coordinate.
    q_data = q_date.assign_coords(time=pd.to_datetime(date_str))
    
    # Select streamflow data for the relevant reaches and rename 'feature_id' to 'reachid'.
    reach_q = q_date.streamflow.sel(feature_id=reaches).rename({'feature_id':'reachid'})

    # Extract polynomial coefficients for the current image.
    reach_coefs = extract_coefs(img, mode_names)

    sim_pcs_list = [] # List to store simulated principal components.

    # Loop over each reach/mode and create a simulated temporal principal component (PC) image.
    for i in range(len(reach_q)):
        reach_q_img = ee.Image(float(reach_q[i])) # Convert streamflow value to an Earth Engine Image.
        # Apply the polynomial coefficients to the streamflow value to simulate the PC.
        reach_sim_pcs = reach_q_img.polynomial(reach_coefs.slice(0,i,i+1).project([1]).toList())
        sim_pcs_list.append(reach_sim_pcs)

    # If no reaches are found, raise an error.
    if len(sim_pcs_list) == 0:
        raise ValueError('No reaches found for this watershed.')

    # Concatenate the simulated principal components into a single Earth Engine Image.
    sim_pcs = ee.Image.cat([sim_pcs_list])

    # Select spatial modes and center image from the FIER collection image.
    modes = img.select(['mode_.*'])
    center = img.select(['center'])

    # Reconstruct the water fraction image using spatial modes, simulated PCs, and the center image.
    reconstructed = (
        modes.toArray()
        .arrayDotProduct(sim_pcs.toArray())
        .add(center)
    )

    # Add the reconstructed image to the list, masked by the current HUC8 watershed.
    imgs.append(reconstructed.updateMask(huc8Id_image.eq(hydroid)))

# Define properties for the final predicted image.
properties = {
    'system:time_start': ee.Date(date_str).millis(),
    'system:time_end': ee.Date(date_str).advance(86399,'seconds').millis(), # End of the day.
    'forecast': FORECAST,
    'forecast_offset': forecast_offset_lookup[FORECAST],
    'reference_date': ee.Date(reference_date.strftime('%Y-%m-%dT00:00:00')).millis(),
    'corrected': 0, # Flag indicating if the pseudoforecast has been corrected (0 = no).
    'buffer_size': 1 # Buffer size used for this pseudoforecast.
}

# Mosaic all reconstructed images and set the defined properties.
pred = ee.ImageCollection.fromImages(imgs).mosaic().set(properties)

# Define the export region as the union of all HUC8 geometries in the FIER collection.
export_region = fier_collection.map(lambda x: ee.Feature(x.geometry())).union().geometry().bounds()

# Define the output Earth Engine asset collection path.
OUTCOLLECTION = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_pseudoforecasts_buffer_01km'

# Attempt to create the asset collection. If it already exists, catch the exception and continue.
try: 
    ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, OUTCOLLECTION)
except ee.EEException:
    # Pass if the asset already exists, otherwise re-raise the exception.
    pass

# Export the predicted pseudoforecast image to an Earth Engine asset.
export = ee.batch.Export.image.toAsset(
    image = pred,
    assetId=f'{OUTCOLLECTION}/fier{date_str.replace("-", "")}_{FORECAST}',
    description=f'fier_pseudoforecast_{date_str.replace("-", "")}_{FORECAST}',
    region = export_region, 
    scale=img.projection().nominalScale().getInfo(), # Nominal scale of the image.
    maxPixels=1e13, # Maximum number of pixels to export.
)
export.start() # Start the export task.
