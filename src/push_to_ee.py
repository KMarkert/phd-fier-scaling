import ee
from google.cloud import storage
import json
from pathlib import Path

# Initialize the Earth Engine API with a specific project.
ee.Initialize(
    project='byu-hydroinformatics-gee'
)

# Define the Google Cloud Storage bucket name where GeoTIFFs are stored.
BUCKET = 'kmarkert_phd_research'

# Define the streamflow model used (e.g., 'nwm' for National Water Model).
streamflow = 'nwm'

# Initialize Google Cloud Storage client and get the specified bucket.
client = storage.Client()
bucket = client.get_bucket(BUCKET)

# Define the input directory where GeoTIFFs are located.
indir = Path('../geotiffs')

# Iterate over different buffer sizes (including 'baseline') to push data to Earth Engine.
for buffer in ['baseline',0, 1, 2, 5, 10, 20, 50]:

    # Define the output Earth Engine asset collection path based on buffer size and streamflow model.
    if isinstance(buffer, str):
        OUTCOLLECTION = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/{streamflow}/fier_scaling_{buffer}'
    else:
        OUTCOLLECTION = f'projects/byu-hydroinformatics-gee/assets/kmarkert/fier/{streamflow}/fier_scaling_buffer_{buffer:02d}km'

    # Attempt to create the Earth Engine asset collection.
    # If it already exists, print a message; otherwise, re-raise any other exceptions.
    try: 
        ee.data.createAsset({'type': 'IMAGE_COLLECTION'}, OUTCOLLECTION)
    except ee.EEException as e:
        if 'Cannot overwrite asset' in str(e):
            print(f'{OUTCOLLECTION} already exists.')
        else:
            print(e)

    # Define the directory where spatial modes GeoTIFFs are located for the current buffer.
    if isinstance(buffer, str):
        modes_data = (indir / f'{buffer}' /'modes').glob('*nwm*.tif')
    else:
        modes_data = (indir / f'buffer_{buffer:02d}km' /'modes').glob('*nwm*.tif')

    # Iterate over each spatial mode GeoTIFF file.
    for mode in modes_data:
        # Construct the path to the corresponding JSON properties file.
        json_file = str(mode.with_suffix('.json')).replace('spatial_modes', 'properties').replace('modes', 'jsons')
        
        # Load properties from the JSON file.
        with open(json_file) as f:
            properties = json.load(f)

        # Extract mode names from the properties dictionary.
        mode_names_ = []
        for key in list(properties.keys()):
            if key.startswith('mode_'):
                mode_names_.append('_'.join(key.split('_')[:2]))

        mode_names = sorted(list(set(mode_names_)))

        # Upload the mode GeoTIFF data to the Google Cloud Storage bucket.
        if isinstance(buffer, str):
            blob = bucket.blob(f'fier_{streamflow}/fier_scaling_{buffer}/{mode.name}')
        else:
            blob = bucket.blob(f'fier_{streamflow}/fier_scaling_buffer_{buffer:02d}km/{mode.name}')
        blob.upload_from_filename(mode)

        # Create the manifest for Earth Engine asset ingestion.
        manifest = dict()
        manifest['properties'] = properties # Include all extracted properties.
        manifest['name'] = OUTCOLLECTION + '/' + mode.stem # Define the asset ID.

        # Specify the source URI for the GeoTIFF in Google Cloud Storage.
        if isinstance(buffer, str):
            sources = [{'uris': [f'gs://{BUCKET}/fier_{streamflow}/fier_scaling_{buffer}/{mode.name}']}]
        else:
            sources = [{'uris': [f'gs://{BUCKET}/fier_{streamflow}/fier_scaling_buffer_{buffer:02d}km/{mode.name}']}]
        manifest['tilesets'] = [{'sources':sources}]

        # Define band information for the asset.
        # 'center' is the first band (index 0).
        band_manifest = [
            {
                'id': 'center',
                'tileset_band_index': 0
            }
        ]

        # Add bands for each spatial mode.
        for i,name in enumerate(mode_names):
            band_manifest.append(
                {
                    'id': name,
                    'tileset_band_index': i+1 # Modes start from index 1.
                }
            )

        manifest['bands'] = band_manifest

        # Start the Earth Engine asset ingestion task.
        ingestion_task = ee.data.newTaskId()
        asset_ingest = ee.data.startIngestion(ingestion_task[0], manifest)
        print(f'Ingestion task {asset_ingest["id"]} for {mode.name} started.')
