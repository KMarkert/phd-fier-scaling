from pathlib import Path
import HydroErr as he
import numpy as np
import scipy.stats as stats
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt

# Define the directory containing REOF (Reconstructed Empirical Orthogonal Functions) datasets for the baseline.
reof_dir = Path(f'/Users/kelmarkert/Documents/byu/phd/fierarama/reof_ds/baseline/')
# Get the first NWM-related REOF file in the directory.
reof_file = list(reof_dir.glob('*nwm.nc'))[0]

# Open the REOF dataset.
ds = xr.open_dataset(reof_file)

# Select data for a specific time slice (2012-01-01 to 2020-12-31).
ds = ds.sel(time=slice('2012-01-01','2020-12-31'))

# Extract relevant variables from the dataset.
reachids = ds['reachid'] # Reach IDs associated with each mode.
modes = ds['mode'] # Mode numbers.
fit_metrics = ds['fit_metrics'] # Metrics related to the fit of the regression.
coefs = ds['coefficients'] # Polynomial coefficients for the regression.

time = ds['time'] # Time dimension.
hydro_var = ds['hydro_var'] # Hydrological variable (e.g., streamflow).
temporal_modes = ds['temporal_modes'] # Temporal principal components (RTPCs).
spatial_modes = ds['spatial_modes'] # Spatial modes (RSMs).

# Define the colormap for spatial mode plots.
cmap = mpl.cm.RdBu_r

# Create a figure and a grid of subplots for visualizing fit metrics.
# The gridspec_kw argument sets custom height ratios for rows.
f, ax = plt.subplots(4, len(modes), figsize=(12, 10), gridspec_kw={'height_ratios': [2, 0.1, 1, 1]})

# Iterate through rows (i) and columns (j) of the subplot grid.
for i in range(4):    
    for j in range(len(modes)):

        if i == 0:
            # Plot spatial modes (RSMs).
            spatial_modes.isel(mode=j).fillna(0).plot(ax=ax[i, j], cmap=cmap, vmin=-0.01, vmax=0.01, add_colorbar=False)
            ax[i, j].set_title('') # Clear default title.
            ax[i, j].set_title(f'RSM-{int(modes.isel(mode=j).values):02d}', loc='left') # Set custom title.
            ax[i, j].set_ylabel('Latitude')
            ax[i, j].set_xlabel('Longitude')

        if i == 1:
            # Turn off axes for this row (used for spacing).
            ax[i, j].axis('off')

        if i == 2:
            # Plot normalized temporal principal components (RTPCs) and NWM streamflow.
            ts_norm = stats.zscore(temporal_modes.isel(mode=j), nan_policy='omit') # Z-score normalize RTPCs.
            hv_norm = stats.zscore(hydro_var.isel(mode=j), nan_policy='omit') # Z-score normalize streamflow.
            ax[i, j].plot(time.values, ts_norm, label='RTPCs')
            ax[i, j].plot(time.values, hv_norm, label='NWM Streamflow')
            ax[i, j].set_title(f'RTPC-{int(modes.isel(mode=j).values):02d}', loc='left') # Set custom title.

            ax[i, j].set_ylabel('Normalized Value')
            ax[i, j].set_xlabel('Date')

            # Set custom x-axis tick labels for dates.
            ax[i,j].set_xticklabels(['','2012','','2014','','2016','','2018','','2020',''])
            if j == 0:
                ax[i, j].legend() # Add legend only to the first subplot in this row.
            

        if i == 3:
            # Plot the regression fit between NWM streamflow and RTPCs.
            hv_mode = hydro_var.isel(mode=j) # Streamflow for the current mode.
            p = np.polynomial.Polynomial(coefs.isel(mode=j).values) # Create a polynomial object from coefficients.
            x = np.linspace(hv_mode.min(), hv_mode.max(), 500) # Generate x-values for plotting the fitted curve.
            y = p(x) # Calculate y-values (fitted RTPCs) for the generated x-values.
            y_ = p(hv_mode) # Calculate fitted RTPCs for the actual streamflow values.

            mode_n = int(modes.isel(mode=j).values) # Get the current mode number.

            nse = he.nse(y_, temporal_modes.isel(mode=j)) # Calculate Nash-Sutcliffe Efficiency (NSE).
            ax[i, j].plot(hydro_var.isel(mode=j), temporal_modes.isel(mode=j), 'C0o', alpha=0.7, label='Data') # Plot scatter data.
            ax[i, j].plot(x, y, 'r', ls='--', label = 'Fitted model\n(NSE: {:.2f})'.format(nse)) # Plot fitted curve.
            ax[i, j].set_xlabel('NWM Streamflow [cms]')
            ax[i, j].set_ylabel(f'RTPC-{mode_n:02d}')
            ax[i, j].legend() # Add legend.

# Adjust subplot parameters to optimize layout.
plt.subplots_adjust(hspace=0.05, bottom=0.1, top=0.95)

# Add a colorbar for the spatial modes.
cax = f.add_axes([0.3,0.53,0.4,0.015]) # Define colorbar axis position.

norm = mpl.colors.Normalize(vmin=-0.01, vmax=0.01) # Normalize colorbar values.

cb = f.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=cax, orientation='horizontal',
             ticks=[-0.01, -0.005, 0, 0.005, 0.01]) # Set colorbar ticks.

cb.ax.set_xticklabels(['-0.01', '-0.005',  '0', '0.005', '0.01']) # Set custom tick labels.

plt.tight_layout() # Adjust layout to prevent labels from overlapping.
plt.savefig('figures/baseline_reof_fit_metrics.png',dpi=300) # Save the figure.
