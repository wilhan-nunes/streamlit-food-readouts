
# Food Biomarkers Readout Analysis

This project processes food biomarkers and generates metadata using input files and workflows. It includes a Streamlit app for an interactive interface and a CLI script for batch processing.

## Requirements

Install the required Python packages using:

```bash
pip install -r requirements.txt
```

## Running the Streamlit App

To launch the Streamlit app, use the following command:

```bash
bash run_app.sh
```

### App Features

1. **Inputs**:
   - Enter the Task ID for the Library Search Workflow.
   - Upload a Sample Feature Table file or provide a Task ID to fetch it.
2. **Processing**:
   - Click the "Run Analysis" button to process the data.
3. **Results**:
   - View the processed food metadata in a table.
   - Download the processed data as a `.tsv` file.

## Running the Script from the CLI

You can also process the data using the CLI script:

```bash
python script.py --biomarkers_file <path_to_biomarkers_file> \
                 --lib_search_file <path_to_lib_search_file> \
                 --metadata_file <path_to_metadata_file> \
                 --sample_quant_table_file <path_to_sample_feature_table_file> \
                 --output_dir <output_directory>
```

### Example Command

```bash
python script.py --biomarkers_file data/Biomarkers_level5_FC5_VIP6.csv \
                 --lib_search_file input_test_files/librarymatches_all_analog.tsv \
                 --metadata_file data/gnps_metadata_ming.tsv \
                 --sample_quant_table_file input_test_files/MSV82493_iimn_gnps_quant.csv
```

The processed metadata will be saved in the specified output directory.

## Fetching Files with Task IDs

The `utils.py` module provides a `fetch_file` function to download files from GNPS using Task IDs. This is integrated into the Streamlit app for automated file retrieval.

## Output

The processed metadata is saved as `food_metadata.csv` in the specified output directory or displayed in the app.

