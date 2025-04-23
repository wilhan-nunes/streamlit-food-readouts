import os

import pandas as pd
import streamlit as st

from script import process_food_biomarkers
from utils import fetch_file

# cute badges
BADGE_LIBRARY_ = ":green-badge[Library]"
UPLOAD_QUANT_TABLE_ = ":blue-badge[Upload Quant Table]"
BADGE_QUANT_TABLE_ = ":orange-badge[Quant Table]"

# Streamlit app title
st.title("Food Biomarkers Analysis")

#defining query params to populate input fields
query_params = st.query_params
lib_task_id = query_params.get('lib_task_id', '')
quant_task_id = query_params.get('quant_task_id', '')

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")

    biomarkers_folder = 'data/biomarker_tables'
    files = os.listdir(biomarkers_folder)
    BIOMARKERS_FILES = [f for f in files if f.endswith(('.csv', '.tsv'))]

    selected_biomarkers_file = st.selectbox(
        "Select Biomarkers File",
        BIOMARKERS_FILES,
        help="Choose the biomarkers file to use for the analysis."
    )

    lib_search_task_id = st.text_input(f"{BADGE_LIBRARY_} Library Search Workflow Task ID (GNPS2)",
                                       help="Enter the Task ID from a Library Search Workflow to retrieve the library search results.",
                                       placeholder='enter task ID...',
                                       value=lib_task_id)
    if not lib_search_task_id:
        st.warning("Please enter a Task ID from a Library Search Workflow to proceed.", )

    quant_table_task_id = st.text_input(f"{BADGE_QUANT_TABLE_} FBMN Workflow Task ID (GNPS2)",
                                        help="Enter the Task ID from a FBMN Workflow to retrieve the quant table.",
                                        placeholder='enter task ID...',
                                        value=quant_task_id)

    sample_feature_table_file = st.file_uploader(f"{UPLOAD_QUANT_TABLE_} Upload Sample Feature Table",
                                                 type=["csv", "tsv"])

    if sample_feature_table_file:
        st.success(f"{UPLOAD_QUANT_TABLE_}  will be used for analysis.")
    elif quant_table_task_id:
        st.success(f"No file uploaded. {BADGE_QUANT_TABLE_} task ID will be used to fetch the quant table.")
    else:
        st.warning(f"Please {UPLOAD_QUANT_TABLE_} or provide a Task ID  for {BADGE_QUANT_TABLE_}")

    run_analysis = st.button("Run Analysis", help="Click to start the analysis with the provided inputs.",
                             use_container_width=True)

# Static file paths
METADATA_FILE = "data/gnps_metadata_ming.tsv"

# Process files when task ID and sample feature table are provided
if run_analysis:
    try:
        # Retrieve lib_search using the task ID
        with st.spinner("Downloading library result table..."):
            lib_search = fetch_file(lib_search_task_id.strip(), "merged_results.tsv", type="library_search_table")
            st.success("Library result table downloaded successfully!")

        with st.spinner("Downloading FBMN Quant table from task ID..."):
            if sample_feature_table_file is None:
                sample_feature_table_file = fetch_file(quant_table_task_id.strip(), "quant_table.csv",
                                                       type="quant_table")
                st.success(f"Quant table downloaded successfully from task {quant_table_task_id}!", icon="🔗")
            else:
                st.success("Sample Feature Table loaded from uploaded file successfully!", icon="📂")

        # Load user-uploaded sample feature table
        sample_feature_table_df = pd.read_csv(sample_feature_table_file, sep=None, engine='python')

        # Process data
        with st.spinner("Processing data..."):
            biomarker_filepath = os.path.join(biomarkers_folder, selected_biomarkers_file)
            process_food_biomarkers(biomarker_filepath, lib_search, METADATA_FILE, sample_feature_table_df)
            st.success("Data processed successfully!")

        # Load and display the resulting table
        result_file = "./output/food_metadata.csv"
        result_data = pd.read_csv(result_file)
        st.write("### Processed Food Metadata")
        st.dataframe(result_data)

        # Download option
        st.download_button(
            label="Download Processed Data",
            data=result_data.to_csv(sep='\t', index=False),
            file_name="food_metadata.tsv",
            mime="text/csv"
        )
    except Exception as e:
        st.error(f"An error occurred: {e}")
        # raise
else:
    st.info(
        ":information_source: Please, provide the inputs, then click Run Analysis.")
