import ee
import numpy as np
from pathlib import Path
import xarray as xr
import logging
import xee
from dask.diagnostics import ProgressBar

# Initialize the Earth Engine API with a specific project and high-volume API URL.
ee.Initialize(
    project='byu-hydroinformatics-gee',
    opt_url=ee.data.HIGH_VOLUME_API_BASE_URL
)

# Define the area of interest (AOI) as the Mississippi basin.
ms_aoi = ee.Feature(
    ee.FeatureCollection('projects/byu-hydroinformatics-gee/assets/kmarkert/fier/mississippi_aoi')
    .first()
)

# Set the name for the baseline dataset.
name = 'baseline'
print(name)

# Define the output path for the NetCDF file for the baseline water fraction data.
outpath = Path(f'../water_fractions/{name}/water_fraction_{name}.nc')

# Create the output directory if it doesn't exist.
if not outpath.parent.exists():
    outpath.parent.mkdir(parents=True)

# Load the HUC8 ID image from Earth Engine.
huc8_ids = ee.Image('projects/byu-hydroinformatics-gee/assets/NOAA_FIER/huc8Id_image')

# Load the NOAA JPSS floods image collection from Earth Engine.
noaa_jpss_floods = ee.ImageCollection('projects/byu-hydroinformatics-gee/assets/noaa_jpss_floods')

def to_water_fraction(img):
    """
    Converts the VIIRS water detection image to actual water fraction units [0-100].

    Args:
        img (ee.Image): An Earth Engine image from the NOAA JPSS floods collection.

    Returns:
        ee.Image: An Earth Engine image representing water fraction (0-100),
                  with invalid pixels masked and time_start property set.
    """
    # Define a mask for valid water detection values.
    valid_mask = img.gte(99).Or(img.eq(16).Or(img.eq(17)))  # Define valid mask
    # Permanent water (value 99) is set to 100% water fraction.
    water_permanent = img.eq(99).multiply(100)  # Permanent water
    # Water fraction for values >= 100 is calculated by subtracting 100.
    # Unmask with 0 to ensure non-water areas are zero.
    water_fraction = img.updateMask(img.gte(100)).subtract(100).unmask(0)  # Water fraction

    return (
        water_permanent.add(water_fraction)
        .updateMask(valid_mask)
        .unmask(255) # Unmask with 255 for no data, to be handled by _FillValue in encoding.
        .float()
        .set({
            'system:time_start': img.date().millis(),  # Set the time start property
        })
    )

# Apply the `to_water_fraction` function over the image collection to get water fractions.
water_fractions = noaa_jpss_floods.map(to_water_fraction)

# Get the projection information (scale and CRS) from the first image in the collection.
proj = water_fractions.first().projection().getInfo()
scale = proj['transform'][0]
crs = proj['crs']

# Open the water fractions dataset using xarray with Earth Engine as the engine.
# Specify the CRS, scale, and geometry for data extraction.
ds = xr.open_dataset(
    water_fractions,
    engine='ee',
    crs=crs,
    scale=scale,
    geometry=ms_aoi.geometry(),
)
# Chunk the dataset for Dask processing and rename the 'water_detection' variable.
ds = ds.chunk({"time": 10, "lon": 512, "lat": 512}).rename({'water_detection': 'water_fraction'})

# Open the Huc8ID mask dataset using xarray with Earth Engine as the engine.
mask_ds = xr.open_dataset(
    ee.ImageCollection([huc8_ids]),
    engine='ee',
    crs=crs,
    scale=scale,
    geometry=ms_aoi.geometry(),
)

# Merge the water fraction dataset with the HUC8 ID mask.
# Transpose dimensions to 'time', 'lat', 'lon' for consistency.
out_ds = xr.merge([ds, mask_ds.huc8Id.isel(time=0)])
out_ds = out_ds.transpose('time', 'lat', 'lon')

# Set the global attributes for the output dataset.
out_ds.attrs = {
    'huc_level': 8,
    'huc_id': name,
    'crs': crs,
    'scale': scale,
}

# Define the encoding settings for the NetCDF file, including compression and fill values.
encoding = {
    "water_fraction": {
        "dtype": "uint8", 
        "zlib": True,
        "complevel": 3,
        "_FillValue": 255,
    },
    "huc8Id": {
        "dtype": "uint32", 
        "zlib": True,
        "complevel": 9,
        "_FillValue": 0,
    },
}

# Write the dataset to a NetCDF file with a progress bar.
with ProgressBar():
    out_ds.to_netcdf(outpath, encoding=encoding)
