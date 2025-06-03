import ee
import numpy as np
from pathlib import Path
import xarray as xr

def remove_duplicates(nested_list):
  """
  Removes duplicate datetime objects from a nested list of numpy.datetime64 objects.

  Args:
    nested_list (list): A nested list where inner lists contain numpy.datetime64 objects.

  Returns:
    list: A new flat list with unique datetime objects, sorted.
  """
  # Flatten the nested list into a single list.
  flat_list = [item for sublist in nested_list for item in sublist]

  # Convert to a set to automatically remove duplicate entries.
  unique_dates = set(flat_list)

  # Convert back to a list and sort it.
  return sorted(list(unique_dates))

def filter_poor_observations(ds, variable, threshold=0.999):
    """
    Filters an xarray Dataset to remove time steps with poor observations.
    A time step is considered to have poor observations if the fraction of finite
    values for the specified variable falls below a given threshold.

    Args:
        ds (xr.Dataset): The input xarray Dataset.
        variable (str): The name of the variable (e.g., 'water_fraction') to check for finite values.
        threshold (float, optional): The minimum fraction of finite values required to keep a time step.
                                     Defaults to 0.999.

    Returns:
        xr.Dataset: The filtered xarray Dataset containing only time steps with sufficient observations.
    """
    da = ds[variable]
    # Count the number of finite values along the longitude and latitude dimensions.
    x = np.isfinite(da).sum(dim=["lon", "lat"])
    # Calculate the fraction of finite values relative to the total number of pixels.
    y = x / (da.shape[1] * da.shape[2])
    # Identify dates where the fraction of finite values is above the threshold and drop NaNs.
    keep_dates = y.where(y > threshold).dropna(dim="time").time
    # Select and return the dataset for the identified dates.
    return ds.sel(time=keep_dates)

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

# Load the HUC8 ID image from Earth Engine.
huc8Id_image = ee.Image('projects/byu-hydroinformatics-gee/assets/NOAA_FIER/huc8Id_image')

# Load NWM streamflow data, rename 'streamflow' to 'hydro_var'.
q = xr.open_dataset('../nwm_streamflow/nwm_streamflow_daily.nc').rename({'streamflow':'hydro_var'})

# Load the FIER scaling image collection for the baseline.
fier_collection = ee.ImageCollection(f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/fier_scaling_baseline')

# Define the directory containing water fraction datasets for the baseline.
waterfraction_dir = Path(f'/Users/kelmarkert/Documents/byu/phd/fierarama/water_fractions/baseline/')

# Define the directory containing REOF (Reconstructed Empirical Orthogonal Functions) datasets for the baseline.
reof_dir = Path(f'/Users/kelmarkert/Documents/byu/phd/fierarama/reof_ds/baseline/')
# Get a list of all NWM-related NetCDF files in the REOF directory.
reofs = reof_dir.glob('*nwm.nc')

times = [] # List to store all valid observation dates.
train_times = [] # List to store dates used for training the FIER model.

# Collect valid observation dates and training dates from REOF and water fraction files.
for reof in reofs:
    # Open the corresponding water fraction dataset.
    wf_ds = xr.open_dataset(waterfraction_dir / str(reof.name).replace('reof_','').replace('_nwm',''))

    # Open the REOF dataset.
    ds = xr.open_dataset(reof)
    
    # Filter water fraction observations for a specific time slice and threshold, then extract dates.
    ds_times = filter_poor_observations(wf_ds.sel(time=slice('2012-01-01','2020-12-31')),'water_fraction',threshold=0.9).time.values
    times.append(ds_times)

    # Extract training dates from the REOF dataset.
    train_dates = ds.time.values
    train_times.append(train_dates)

# Get unique and sorted lists of all observation dates and training dates.
dates = sorted(remove_duplicates(times))
train_dates = sorted(remove_duplicates(train_times))

print(f"Total images to export: {len(dates)}")
print(f"Total training images: {len(train_dates)}")

# Iterate over each date to generate and export predictions.
for date in dates:
    date_str = str(date)[:10] # Convert numpy datetime to string format 'YYYY-MM-DD'.
    print('Exporting:', date_str)

    # Select streamflow data for the current date.
    q_date = q.hydro_var.sel(time=date_str)

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

    # For baseline, there's typically only one image in the fier_collection.
    img = fier_collection.first()

    # Get band names and remove 'center' to get only mode names.
    mode_names = img.bandNames().remove('center')

    # Extract reach IDs associated with the modes.
    reaches_ = extract_reaches(img, mode_names)
    reaches = reaches_.getInfo()

    # Select streamflow data for the relevant reaches and rename 'feature_id' to 'reachid'.
    # This handles cases where some feature_ids might not be present in q_date.
    domain_qs = []
    for i in reaches:
        try:
            domain_qs.append(q_date.sel(feature_id=i))
        except KeyError:
            pass
    reach_q = xr.concat(domain_qs, dim='feature_id').rename({'feature_id':'reachid'})

    # Extract polynomial coefficients for the current image.
    reach_coefs = extract_coefs(img, mode_names)

    sim_pcs_list = [] # List to store simulated principal components.

    # Loop over each reach/mode and create a simulated temporal principal component (PC) image.
    for i in range(len(reach_q)):
        reach_q_img = ee.Image(float(reach_q[i])) # Convert streamflow value to an Earth Engine Image.
        # Apply the polynomial coefficients to the streamflow value to simulate the PC.
        reach_sim_pcs = reach_q_img.polynomial(reach_coefs.slice(0,i,i+1).project([1]).toList())
        sim_pcs_list.append(reach_sim_pcs)
    
    # If no simulated PCs are generated, skip this date.
    if len(sim_pcs_list) == 0:
        continue

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

    # Set the system:time_start property for the reconstructed image.
    pred = reconstructed.set('system:time_start', ee.Date(date_str).millis())

    # Define the export region as the union of all HUC8 geometries in the FIER collection.
    export_region = fier_collection.map(lambda x: ee.Feature(x.geometry())).union().geometry().bounds()

    # Define the output Earth Engine asset collection path.
    OUTCOLLECTION = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/nwm/predictions/fier_baseline'

    # Attempt to create the asset collection. If it already exists, catch the exception and continue.
    try: 
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, OUTCOLLECTION)
    except ee.EEException:
        # Pass if the asset already exists, otherwise re-raise the exception.
        pass

    # Determine if the current date was part of the training set.
    if date in train_dates:
        train = 1
    else:
        train = 0

    # Export the predicted image to an Earth Engine asset.
    export = ee.batch.Export.image.toAsset(
        image = pred.set('train',train), # Set the 'train' property.
        assetId=f'{OUTCOLLECTION}/fier_test_{date_str.replace("-", "")}',
        description=f'fier_test_{date_str.replace("-", "")}',
        region = export_region, 
        scale=img.projection().nominalScale().getInfo(), # Nominal scale of the image.
        maxPixels=1e13, # Maximum number of pixels to export.
    )
    export.start() # Start the export task.
