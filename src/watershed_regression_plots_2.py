from pathlib import Path
import ee
import HydroErr as he
import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.stats as stats
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
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

# Compute watershed features as a GeoPandas DataFrame for local processing.
watershed_df = ee.data.computeFeatures(dict(
    expression = watersheds,
    fileFormat = 'GEOPANDAS_GEODATAFRAME'
))

# Define a list of buffer sizes to analyze.
buffers = [0, 1, 2, 5, 10, 20, 50]

buffer_dfs = {} # Dictionary to store DataFrames of metrics for each buffer size.

# Iterate over each buffer size to calculate and store regression metrics.
for buffer in buffers:
    # Define the directory containing REOF datasets for the current buffer.
    reof_dir = Path(f'/Users/kelmarkert/Documents/byu/phd/fierarama/reof_ds/buffer_{buffer:02d}km/')
    # Get a list of all NWM-related NetCDF files in the REOF directory.
    reof_files = list(reof_dir.glob('*nwm.nc'))

    wids = [] # List to store watershed IDs.
    wcoef = [] # List to store average fit metrics (Pearson's R).
    wnse = [] # List to store average Nash-Sutcliffe Efficiency (NSE).
    wnmodes = [] # List to store number of modes.

    # Iterate over each REOF file (representing a watershed).
    for reof_file in reof_files:
        # Open the REOF dataset.
        ds = xr.open_dataset(reof_file)

        # Extract watershed ID from the filename.
        watershed_id = reof_file.name.split('_')[3]

        # Select data for a specific time slice (2012-01-01 to 2020-12-31).
        ds = ds.sel(time=slice('2012-01-01','2020-12-31'))

        # Extract relevant variables.
        modes = ds['mode']
        fit_metrics = ds['fit_metrics']
        coefs = ds['coefficients']
        hydro_var = ds['hydro_var']
        temporal_modes = ds['temporal_modes']

        # Calculate the average fit metric (Pearson's R) across all modes for the watershed.
        avg_coefs = float(fit_metrics.mean(dim='mode').values)

        nses = [] # List to store NSE for each mode.
        # Calculate NSE for each mode.
        for mode in modes:
            hv_mode = hydro_var.sel(mode=mode) # Streamflow for the current mode.
            p = np.polynomial.Polynomial(coefs.sel(mode=mode).values) # Polynomial fit.
            y = p(hv_mode) # Predicted temporal PCs.
            nse = he.nse(y, temporal_modes.sel(mode=mode)) # Calculate NSE.
            nses.append(nse)

        # Calculate the average NSE across all modes for the watershed.
        avg_nse = np.mean(nses)

        # Append calculated metrics to lists.
        wids.append(watershed_id)
        wcoef.append(avg_coefs)
        wnse.append(avg_nse)
        wnmodes.append(len(modes))

    # Create a Pandas DataFrame from the collected watershed metrics.
    reof_df = pd.DataFrame({'watershed_id':wids, 'avg_metrics':wcoef, 'avg_nse':wnse, 'n_modes':wnmodes})

    # Merge the watershed metrics DataFrame with the watershed GeoDataFrame.
    watershed_df_metrics = watershed_df.merge(reof_df, left_on='huc8', right_on='watershed_id')

    # Store the merged DataFrame in the dictionary, keyed by buffer size.
    buffer_dfs[buffer] = watershed_df_metrics

# Concatenate metrics from all buffer sizes into single NumPy arrays for determining global min/max.
for i, buffer in enumerate(buffers):
    if i == 0:
        modes_arr = np.array(buffer_dfs[buffer]['n_modes'].values)
        metrics_arr = np.array(buffer_dfs[buffer]['avg_metrics'].values)
        nse_arr = np.array(buffer_dfs[buffer]['avg_nse'].values)
    else:
        modes_arr = np.concatenate((modes_arr, buffer_dfs[buffer]['n_modes'].values))
        metrics_arr = np.concatenate((metrics_arr, buffer_dfs[buffer]['avg_metrics'].values))
        nse_arr = np.concatenate((nse_arr, buffer_dfs[buffer]['avg_nse'].values))

# Determine global min/max values for consistent color scaling across plots.
min_modes = np.min(modes_arr, axis=0)
max_modes = np.max(modes_arr, axis=0)
min_metrics = 0.6 # Hardcoded minimum for Pearson's R.
max_metrics = 0.85 # Hardcoded maximum for Pearson's R.
min_nse = 0.35 # Hardcoded minimum for NSE.
max_nse = 0.75 # Hardcoded maximum for NSE.

# Define height ratios for subplots.
hratios = [1,1,1,1,0.3] # This seems to be a copy-paste error from _1.py, should be adjusted for _2.py

# Create a figure and a grid of subplots for visualizing watershed metrics for the remaining buffers.
# This script plots buffers from index 4 onwards, plus buffer 0 for comparison.
f, ax = plt.subplots(len(buffers[4:])+2, 3, figsize=(10, 12), gridspec_kw={'height_ratios': hratios})

# Iterate through buffer sizes, starting with buffer 0, then buffers from index 4 onwards.
for i, buffer in enumerate([0,] + buffers[4:]):

    # Plot number of RSMs.
    buffer_dfs[buffer].plot(column='n_modes', legend=False, vmin=min_modes, vmax=max_modes, ax=ax[i,0])
    # Plot average Pearson's R.
    buffer_dfs[buffer].plot(column='avg_metrics', legend=False, vmin=min_metrics, vmax=max_metrics, ax=ax[i,1])
    # Plot average NSE.
    buffer_dfs[buffer].plot(column='avg_nse', legend=False, vmin=min_nse,vmax=max_nse, ax=ax[i, 2])

    # Set y-axis label for each row.
    ax[i, 0].set_ylabel(f'Buffer size {buffer} km\n\nLatitude', fontsize=12)

    # Set x-axis labels only for the last row of plots (before the colorbar row).
    if i == 3: # This condition might need adjustment based on the actual number of rows plotted.
        ax[i, 0].set_xlabel('Longitude', fontsize=12)
        ax[i, 1].set_xlabel('Longitude', fontsize=12)
        ax[i, 2].set_xlabel('Longitude', fontsize=12)

# Turn off axes for the last row (used for colorbars).
ax[-1, 0].axis('off')
ax[-1, 1].axis('off')
ax[-1, 2].axis('off')

# Define titles, min, and max values for colorbars.
titles = ['Number of RSMs', "Average Pearson's R", 'Average NSE']
mins = [min_modes, min_metrics, min_nse]
maxs = [max_modes, max_metrics, max_nse]

# Define positions for colorbar axes.
caxes = [
    f.add_axes([0.12,0.05,0.2,0.01]),
    f.add_axes([0.43,0.05,0.2,0.01]),
    f.add_axes([0.74,0.05,0.2,0.01])
]

# Define the colormap for the metric plots.
cmap = mpl.cm.viridis

# Add colorbars to the figure.
for i in range(3):
    ax[1, i].set_title(titles[i], loc='left') # Set title for the second row of plots (index 1).

    norm = mpl.colors.Normalize(vmin=mins[i], vmax=maxs[i]) # Normalize colorbar values.

    cb = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                cax=caxes[i], orientation='horizontal',
                ticks=[mins[i], maxs[i]],
                label=titles[i]) # Add colorbar with ticks and label.

plt.tight_layout() # Adjust layout to prevent labels from overlapping.

plt.savefig(f'figures/buffer_watershed_metrics_2.png', dpi=300) # Save the figure.

# Print descriptive statistics for each buffer's metrics.
for buffer in buffers:
    print(f"Buffer {buffer} km statistics:")
    print(buffer_dfs[buffer].describe())
    print('\n\n')
