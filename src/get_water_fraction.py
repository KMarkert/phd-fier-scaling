import ee
import numpy as np
from pathlib import Path
import xarray as xr
import logging
import xee # xarray-earthengine extension
from dask.diagnostics import ProgressBar

# Initialize the Earth Engine API with a specific project and high-volume API URL.
ee.Initialize(
    project='byu-hydroinformatics-gee',
    opt_url=ee.data.HIGH_VOLUME_API_BASE_URL
)

# Load the HUC08 watershed boundaries from Earth Engine.
watersheds_tbl = ee.FeatureCollection("USGS/WBD/2017/HUC08")

# Define the Mississippi AOI (Area of Interest) by buffering and bounding a predefined feature.
ms_aoi = ee.Feature(
    ee.FeatureCollection('projects/byu-hydroinformatics-gee/assets/kmarkert/fier/mississippi_aoi')
    .first()
).buffer(50*1000).bounds()

# Filter watersheds to include only those that intersect with the Mississippi AOI.
watersheds = watersheds_tbl.filterBounds(ms_aoi.geometry())

# Get a list of HUC8 IDs for the filtered watersheds.
hydroids = ee.List(watersheds.aggregate_array('huc8')).getInfo()
print(f"Processing {len(hydroids)} watersheds")

# Iterate over each HUC8 ID to process water fraction data.
for hydroid in hydroids:
    print(f'Processing {hydroid}')

    # Filter the watershed feature collection for the current HUC8 ID.
    watershed = watersheds.filter(ee.Filter.eq('huc8',hydroid))

    # Define a list of buffer sizes to process for each watershed.
    buffers = [0, 1, 2, 5, 10, 20, 50]
    for buffer in buffers:
        print(f"  Buffer: {buffer} km")
        buffer_km = buffer
        
        # Define the AOI for the current watershed, with or without a buffer.
        if buffer_km == 0:
            watershed_aoi = watershed.geometry().bounds()
        else:
            watershed_aoi = watershed.geometry().buffer(buffer_km * 1000).bounds()

        # Define the output path for the NetCDF file.
        outpath = Path(f'../water_fractions/buffer_{buffer:02d}km/water_fraction_{hydroid}.nc')
        
        # Skip processing if the output file already exists.
        if outpath.exists():
            continue

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
            valid_mask = img.gte(99).Or(img.eq(16).Or(img.eq(17)))
            # Permanent water (value 99) is set to 100% water fraction.
            water_permanent = img.eq(99).multiply(100)
            # Water fraction for values >= 100 is calculated by subtracting 100.
            # Unmask with 0 to ensure non-water areas are zero.
            water_fraction = img.updateMask(img.gte(100)).subtract(100).unmask(0)

            return (
                water_permanent.add(water_fraction)
                .updateMask(valid_mask)
                .unmask(255) # Unmask with 255 for no data, to be handled by _FillValue in encoding.
                .float()
                .set({
                    'system:time_start':img.date().millis(),
                })
            )

        # Apply the `to_water_fraction` function to the NOAA JPSS floods image collection.
        water_fractions = noaa_jpss_floods.map(to_water_fraction)

        # Get projection information (scale and CRS) from the first image in the collection.
        proj = water_fractions.first().projection().getInfo()
        scale = proj['transform'][0]
        crs = proj['crs']

        # Open the water fractions image collection as an xarray Dataset using xee.
        # Specify the CRS, scale, and geometry for data extraction.
        ds = xr.open_dataset(
            water_fractions,
            engine='ee',
            crs=crs,
            scale = scale,
            geometry=watershed_aoi,
        )
        # Chunk the dataset for Dask processing and rename the 'water_detection' variable.
        ds = ds.chunk({"time": 10, "lon":512, "lat":512}).rename({'water_detection':'water_fraction'})

        # Open the HUC8 ID image as an xarray Dataset using xee for masking.
        mask_ds = xr.open_dataset(
            ee.ImageCollection([huc8_ids]),
            engine='ee',
            crs=crs,
            scale = scale,
            geometry=watershed_aoi,
        )

        # Merge the water fraction dataset with the HUC8 ID mask.
        # Transpose dimensions to 'time', 'lat', 'lon' for consistency.
        out_ds = xr.merge([ds, mask_ds.huc8Id.isel(time=0)])
        out_ds = out_ds.transpose('time','lat','lon')

        # Set global attributes for the output dataset.
        out_ds.attrs = {
            'huc_level': 8,
            'huc_id': hydroid,
            'crs': crs,
            'scale': scale,
        }

        # Define encoding settings for NetCDF export, including compression and fill values.
        encoding = {
            "water_fraction": {
                "dtype": "uint8", # Data type for water fraction.
                "zlib": True, # Enable zlib compression.
                "complevel": 3, # Compression level.
                "_FillValue": 255, # Fill value for masked data.
            },
            "huc8Id": {
                "dtype": "uint32", # Data type for HUC8 ID.
                "zlib": True, # Enable zlib compression.
                "complevel": 9, # Higher compression level for HUC8 ID.
                "_FillValue": 0, # Fill value for HUC8 ID.
            },
        }
        # Write the dataset to a NetCDF file with a progress bar.
        with ProgressBar():
            out_ds.to_netcdf(outpath, encoding=encoding)
