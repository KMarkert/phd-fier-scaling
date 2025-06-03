import geopandas as gpd
import fierpy
import numpy as np
from shapely.geometry import box
import xarray as xr
from pathlib import Path

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
    # Select the data array for the given variable.
    da = ds[variable]
    # Count the number of finite values along the longitude and latitude dimensions.
    x = np.isfinite(da).sum(dim=["lon", "lat"])
    # Calculate the fraction of finite values.
    y = x / (da.shape[1] * da.shape[2])
    # Keep dates where the fraction of finite values is above the threshold.
    keep_dates = y.where(y > threshold).dropna(dim="time").time
    # Return the dataset filtered by the selected dates.
    return ds.sel(time=keep_dates)

# Define the name for the baseline dataset.
name = 'baseline'
print(f"RUNNING {name}")

# Define the directory containing water fraction files for the baseline.
water_fraction_dir = Path(f'../water_fractions/{name}/')

# Get a list of all NetCDF files in the water fraction directory (should be only one for baseline).
water_fraction_inputs = water_fraction_dir.glob('*.nc')

# Read the NWM stream network data and convert to EPSG:4326 coordinate reference system.
streams = gpd.read_file('../nwm_streamflow/nwm_streams.gpkg').to_crs('EPSG:4326')
# Filter streams to include only those with stream order >= 6.
streams = streams.loc[streams['streamOrder'] >= 6]
# Read the NWM streamflow data and rename the variable to 'hydro_var'.
q = xr.open_dataset('../nwm_streamflow/nwm_streamflow_daily.nc').rename({'streamflow':'hydro_var'})

# Iterate over each water fraction input file.
for water_fraction_input in water_fraction_inputs:
    # Define the output path for the processed REOF data.
    outpath = Path(f'../reof_ds/{name}/') / f'reof_{water_fraction_input.stem}_test.nc'

    # Create the output directory if it doesn't exist.
    if not outpath.parent.exists():
        outpath.parent.mkdir(parents=True)

    # Process the file if the output file doesn't already exist.
    if not outpath.exists():
        # Open the water fraction dataset.
        water_fraction = xr.open_dataset(water_fraction_input)
        # Filter out poor observations.
        water_fraction = filter_poor_observations(water_fraction, 'water_fraction')

        # Get the longitude and latitude values.
        lons = water_fraction.lon.values
        lats = water_fraction.lat.values

        # Define the bounding box for the water fraction data.
        xmin, xmax = lons.min(), lons.max()
        ymin, ymax = lats.min(), lats.max()
        bbox = box(xmin, ymin, xmax, ymax)

        # Create a GeoDataFrame for the bounding box.
        extent = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs="EPSG:4326")

        # Select streams that intersect with the bounding box.
        # Drop duplicates based on 'station_id' and reset index.
        domain_streams = gpd.sjoin(streams, extent, predicate='intersects')
        domain_streams = domain_streams.drop_duplicates(subset=['station_id']).reset_index()
        assert domain_streams.station_id.is_unique # Ensure unique station IDs.

        # Select streamflow data for the selected streams.
        domain_qs = []
        for i in domain_streams.station_id.values:
            try:
                domain_qs.append(q.hydro_var.sel(feature_id=i))
            except KeyError: # Catch KeyError if feature_id is not found.
                pass
        # Concatenate streamflow data and rename 'feature_id' to 'reachid'.
        domain_q = xr.concat(domain_qs, dim='feature_id').rename({'feature_id':'reachid'})

        # Perform EOF analysis on the water fraction data.
        eof_ds = fierpy.reof(water_fraction.water_fraction)

        # Find the hydrological modes that correspond to the EOFs.
        # `deoutlier` removes outliers, `threshold` filters modes based on correlation.
        hydro_modes = fierpy.find_hydro_mode(eof_ds, domain_q, deoutlier=True, threshold=0.6)

        # Combine the EOFs with the hydrological modes.
        eof_hydro = fierpy.combine_eof_hydro(eof_ds, domain_q, hydro_modes)

        # Save the combined dataset to a NetCDF file.
        eof_hydro.to_netcdf(outpath)
