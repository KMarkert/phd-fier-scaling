import json
from pathlib import Path
import rioxarray
import xarray as xr

# Define the streamflow model used (e.g., 'nwm' for National Water Model).
streamflow = 'nwm'

# Iterate over different buffer sizes (including 'baseline') to unpack REOF data.
for buffer in ['baseline',0, 1, 2, 5, 10, 20, 50]:

    # Define the directory containing REOF datasets for the current buffer.
    if isinstance(buffer, str):
        reof_dir = Path(f'../reof_ds/{buffer}/')
    else:
        reof_dir = Path(f'../reof_ds/buffer_{buffer:02d}km/')

    # Get a list of input NetCDF files based on the streamflow model.
    if streamflow == 'nwm':      
        inputs = reof_dir.glob('*_nwm.nc')
    elif streamflow =='geoglows':
        inputs = reof_dir.glob('*.nc')

    # Iterate over each input REOF NetCDF file.
    for input in inputs:
        # Open the REOF dataset.
        reof_ds = xr.open_dataset(input)
        # Set the Coordinate Reference System (CRS) to EPSG:4326 and rename dimensions for compatibility with rioxarray.
        reof_ds = reof_ds.rio.write_crs(4326).rename({'lat':'y','lon':'x'})

        # Extract the 'center' band and expand its dimensions to be compatible with modes.
        center = reof_ds.center.expand_dims('mode')
        center = center.assign_coords(mode=['center']) # Assign a mode name 'center'.

        # Extract spatial modes, transpose dimensions to 'mode', 'y', 'x'.
        out_modes = (
            reof_ds
            .spatial_modes
            .transpose('mode','y','x')
        )
        # Assign descriptive mode names (e.g., 'mode_0', 'mode_1').
        out_modes = out_modes.assign_coords(mode=['mode_' + str(i) for i in out_modes['mode'].values])
        
        # Concatenate the 'center' band with the spatial modes.
        final_out = xr.concat([center, out_modes], dim='mode')

        # Define the output directory for GeoTIFF spatial modes.
        if isinstance(buffer, str):
            mode_dir = Path(f'../geotiffs/{buffer}/modes/')
        else:
            mode_dir = Path(f'../geotiffs/buffer_{buffer:02d}km/modes/')
        
        # Create the output directory if it doesn't exist.
        if not mode_dir.exists():
            mode_dir.mkdir(parents=True)

        # Export the combined spatial modes and center band to a GeoTIFF file.
        final_out.to_dataset(dim='mode').rio.to_raster(mode_dir / (input.stem + '_spatial_modes.tif'))

        # Get the names of the spatial modes.
        mode_names = out_modes.mode.values

        properties = dict() # Dictionary to store properties for the JSON file.

        # Store the number of modes.
        properties['n_modes'] = reof_ds.mode.shape[0]
        
        # Store the HUC8 ID (or 'baseline' for the baseline case).
        if isinstance(buffer, str):
            properties['huc8Id']= buffer
        else:
            # Extract HUC8 ID from the input file name.
            properties['huc8Id'] = int(input.stem.split('_')[-2].split('.')[0])

        # Define the output directory for JSON properties files.
        if isinstance(buffer, str):
            json_dir = Path(f'../geotiffs/{buffer}/jsons/')
        else:
            json_dir = Path(f'../geotiffs/buffer_{buffer:02d}km/jsons/')
        
        # Create the JSON output directory if it doesn't exist.
        if not json_dir.exists():
            json_dir.mkdir(parents=True)

        # Store polynomial coefficients and associated reach IDs for each mode in properties.
        for i, mode_name in enumerate(mode_names):
            properties[mode_name+'_coeff_0'] = float(reof_ds.coefficients.isel(mode=i, degree=0))
            properties[mode_name+'_coeff_1'] = float(reof_ds.coefficients.isel(mode=i, degree=1))
            properties[mode_name+'_coeff_2'] = float(reof_ds.coefficients.isel(mode=i, degree=2))

            properties[mode_name+'_reach'] = int(reof_ds.reachid.isel(mode=i))

        # Write the properties dictionary to a JSON file.
        with open(json_dir / (input.stem + '_properties.json'), 'w') as outfile:
            json.dump(properties, outfile)
