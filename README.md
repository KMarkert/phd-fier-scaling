# FIER Scaling for Flood Inundation Mapping

This repository contains Python scripts for scaling Flood Inundation Extent from Remote Sensing (FIER) data using hydrological variables, primarily from the National Water Model (NWM). The scripts facilitate the processing, analysis, and prediction of water inundation extents.

## Project Structure

The `src/` directory contains the core Python scripts:

-   `correct_predictions.py`: Corrects FIER predictions using observed water fraction percentiles.
-   `correct_pseudoforecasts.py`: Corrects FIER pseudoforecasts using observed water fraction percentiles.
-   `export_err_map.py`: Exports error maps (absolute error, squared error, SSIM) between predicted and observed water fractions.
-   `export_stats.py`: Exports statistical summaries of error metrics for different buffer sizes and correction statuses.
-   `fier_fitting.py`: Fits the FIER model by performing REOF analysis on water fraction data and combining it with hydrological modes for various buffer sizes.
-   `fier_fitting_baseline.py`: Fits the FIER model for the baseline scenario (no buffer).
-   `get_nwm_streamflow.py`: Downloads and processes National Water Model (NWM) streamflow data from an S3 bucket.
-   `get_water_fraction.py`: Downloads and processes VIIRS water fraction data for various HUC8 watersheds and buffer sizes from Earth Engine.
-   `get_water_fraction_baseline.py`: Downloads and processes VIIRS water fraction data for the entire Mississippi basin (baseline).
-   `predict_pseudo_forecasts.py`: Generates pseudoforecasts of water inundation using the fitted FIER model and NWM forecast data.
-   `prediction.py`: Generates water inundation predictions using the fitted FIER model and historical NWM streamflow data for various buffer sizes.
-   `prediction_baseline.py`: Generates water inundation predictions for the baseline scenario.
-   `push_to_ee.py`: Pushes processed spatial modes and associated properties (coefficients, reach IDs) as Earth Engine assets.
-   `regression_plots.py`: Generates regression plots showing the relationship between hydrological variables and temporal principal components for the baseline.
-   `unpack_reof.py`: Unpacks REOF analysis results (spatial modes, center, coefficients) into GeoTIFFs and JSON property files.
-   `watershed_regression_plots_1.py`: Generates watershed-level regression plots for various buffer sizes (part 1).
-   `watershed_regression_plots_2.py`: Generates watershed-level regression plots for various buffer sizes (part 2).
-   `watershed_regression_plots_dissertation.py`: Generates comprehensive watershed-level regression plots for dissertation.

## Setup and Dependencies

This project relies heavily on Google Earth Engine (GEE) and Google Cloud Platform (GCP) services, along with several Python libraries.

### 1. Python Environment

It is recommended to use a virtual environment.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate # On Linux/macOS
# venv\Scripts\activate # On Windows
```

### 2. Install Python Packages

Install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

*(Note: A `requirements.txt` file is assumed to exist in the root directory of the project. If not, you will need to create one with the following packages: `earthengine-api`, `google-cloud-bigquery`, `geopandas`, `fierpy`, `numpy`, `shapely`, `xarray`, `s3fs`, `dask`, `rioxarray`, `matplotlib`, `HydroErr`, `scipy`, `pandas`, `xarray-earthengine`)*

### 3. Earth Engine Authentication

You need to authenticate your Earth Engine account.

```bash
earthengine authenticate
```

Follow the instructions in your browser to log in and grant access. Ensure your GEE project (`byu-hydroinformatics-gee` as specified in the scripts) is correctly set up and has the necessary permissions.

### 4. Google Cloud Platform (GCP) Setup

Some scripts interact with Google Cloud Storage and BigQuery.

-   **Google Cloud Storage**: Ensure you have a GCP project and a storage bucket (`kmarkert_phd_research` as specified in `push_to_ee.py`).
-   **BigQuery**: Ensure you have access to the `bigquery-public-data.national_water_model` dataset. The `predict_pseudo_forecasts.py` script uses a personal project (`kmarkert-personal`) for BigQuery client initialization; you may need to change this to your own project ID.
-   **Authentication**: Ensure your environment is authenticated to GCP. If running locally, `gcloud auth application-default login` might be necessary, or set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to point to a service account key file.

## How to Run

The scripts are designed to be run individually or as part of a larger workflow. The typical workflow involves:

1.  **Data Acquisition**:
    *   `get_nwm_streamflow.py`: Downloads NWM streamflow data.
    *   `get_water_fraction.py` / `get_water_fraction_baseline.py`: Downloads VIIRS water fraction data.
2.  **FIER Model Fitting**:
    *   `fier_fitting.py` / `fier_fitting_baseline.py`: Fits the FIER model.
3.  **REOF Unpacking and Earth Engine Push**:
    *   `unpack_reof.py`: Prepares REOF results for Earth Engine.
    *   `push_to_ee.py`: Uploads spatial modes and properties to Earth Engine.
4.  **Prediction/Pseudoforecast Generation**:
    *   `prediction.py` / `prediction_baseline.py`: Generates historical predictions.
    *   `predict_pseudo_forecasts.py`: Generates pseudoforecasts.
5.  **Correction and Error Analysis**:
    *   `correct_predictions.py` / `correct_pseudoforecasts.py`: Corrects predictions/pseudoforecasts.
    *   `export_err_map.py`: Exports error maps.
    *   `export_stats.py`: Exports error statistics.
6.  **Plotting**:
    *   `regression_plots.py`: Generates plots for model fitting.
    *   `watershed_regression_plots_1.py`, `watershed_regression_plots_2.py`, `watershed_regression_plots_dissertation.py`: Generate watershed-level analysis plots.

To run a specific script, navigate to the `src/` directory (or the new `phd-fier-scaling/src/` directory) and execute it using Python:

```bash
cd /Users/kelmarkert/Documents/byu/phd/git/phd-fier-scaling/src/
python3 your_script_name.py
```

**Important Notes:**

-   Many scripts contain hardcoded paths (e.g., `/Users/kelmarkert/Documents/byu/phd/fierarama/`). You will need to **update these paths** to match your local directory structure.
-   Some scripts have commented-out lines for testing or specific runs (e.g., `dates[:1]` in `prediction.py`). Uncomment or modify these as needed for full execution.
-   Ensure you have sufficient permissions for file system operations and Earth Engine/GCP asset management.
