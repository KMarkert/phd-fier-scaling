import numpy as np
import pandas as pd
import s3fs
import xarray as xr
from dask.diagnostics import ProgressBar

# Path to the file containing NWM feature IDs (one ID per line).
# These IDs correspond to specific stream reaches in the National Water Model.
input_ids = '/Users/kelmarkert/Documents/byu/phd/fierarama/nwm_streamflow/nwm_feature_ids.txt'

# Path to the output NetCDF file where the daily streamflow data will be saved.
output_file = '/Users/kelmarkert/Documents/byu/phd/fierarama/nwm_streamflow/nwm_streamflow_daily.nc'

# Read feature IDs from the input file.
with open(input_ids, 'r') as f:
    feature_ids = f.readlines()

# Clean and sort the feature IDs, converting them to integers.
feature_ids = sorted([int(x.replace('\n', '')) for x in feature_ids])

# S3 bucket URI for the NOAA NWM retrospective 3.0 Zarr dataset.
bucket_uri = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr/chrtout.zarr'
# AWS region name where the S3 bucket is located.
region_name = 'us-east-1'

# Initialize S3 filesystem with anonymous access.
s3 = s3fs.S3FileSystem(anon=True, client_kwargs=dict(region_name=region_name))
# Create an S3Map for direct Zarr access.
s3store = s3fs.S3Map(root=bucket_uri, s3=s3, check=False)

# Open the Zarr dataset from S3.
# Drop unnecessary variables to reduce memory footprint and select data from '2012-01-01' onwards.
ds = (
    xr.open_zarr(s3store)
    .drop_vars(['crs', 'order', 'latitude', 'longitude', 'elevation', 'gage_id'])
    .sel(time=slice('2012-01-01', None))
)

# Select the 'streamflow' variable, filter by the specified feature IDs,
# and resample the data to daily mean values.
ds = ds[['streamflow']].sel(feature_id=feature_ids).resample(time='1D').mean()

# Define chunk sizes for the output NetCDF file.
# This helps with efficient reading and writing of large datasets.
out_chunks = {'feature_id': 1, 'time': len(ds.time)}

# Write the dataset to a NetCDF file with specified encoding options.
# 'streamflow' variable is compressed using zlib with compression level 3,
# and chunked for optimized access.
with ProgressBar(): # Display a progress bar during the writing process.
    ds.to_netcdf(
        output_file,
        encoding={
            'streamflow': {
                'dtype': 'float32', # Data type for streamflow.
                'zlib': True, # Enable zlib compression.
                'complevel': 3, # Compression level (1-9, 9 is highest).
                'chunksizes': (1, len(ds.time)) # Chunking strategy for the variable.
            }
        }
    )
