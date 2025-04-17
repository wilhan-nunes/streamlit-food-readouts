import pandas as pd
import streamlit as st

from script import process_food_biomarkers
from utils import fetch_file

# Streamlit app title
st.title("Food Biomarkers Analysis")

# Sidebar inputs
st.sidebar.header("Inputs")

lib_search_task_id = st.sidebar.text_input("Enter Task ID from a Library Search Workflow",
                                           value='34e2b1b692444bf6ae37e71dd137c300')
if not lib_search_task_id:
    st.sidebar.warning("Please enter a Task ID from a Library Search Workflow to proceed.",)

#TODO: verify if the file retrieved contains all necessary columns for the current process_food_biomarkers() function
quant_table_task_id = st.sidebar.text_input("Enter Task ID from a FBMN Workflow",
                                            placeholder="Enter here the task ID for the job from which the quant table should be retrieved.")

sample_feature_table_file = st.sidebar.file_uploader("Upload Sample Feature Table", type=["csv", "tsv"])

if sample_feature_table_file:
    st.sidebar.success("Uploaded Sample Feature Table will be used for analysis.")
elif quant_table_task_id:
    st.sidebar.info("No file uploaded. Task ID will be used to fetch the quant table.")
else:
    st.sidebar.warning("Please upload a Sample Feature Table or provide a Task ID to fetch the quant table.")


run_analysis = st.sidebar.button("Run Analysis", help="Click to start the analysis with the provided inputs.", use_container_width=True)

# Static file paths
BIOMARKERS_FILE = "data/Biomarkers_level5_FC5_VIP6.csv"
METADATA_FILE = "data/gnps_metadata_ming.tsv"

# Process files when task ID and sample feature table are provided
# if lib_search_task_id and sample_feature_table_file and run_analysis:
if run_analysis:
    try:
        # Retrieve lib_search using the task ID
        with st.spinner("Downloading library result table..."):
            lib_search = fetch_file(lib_search_task_id.strip(), "merged_results.tsv", type="library_search_table")
            st.success("Library result table downloaded successfully!")

        with st.spinner("Downloading FBMN Quant table from task ID..."):
            if sample_feature_table_file is None:
                sample_feature_table_file = fetch_file(quant_table_task_id.strip(), "quant_table.csv", type="quant_table")
                st.success(f"Quant table downloaded successfully from task {quant_table_task_id}!", icon="🔗")
            else:
                st.success("Sample Feature Table loaded from uploaded file successfully!", icon="📂")

        # Load user-uploaded sample feature table
        sample_feature_table_df = pd.read_csv(sample_feature_table_file, sep=None, engine='python')

        # Process data
        with st.spinner("Processing data..."):
            process_food_biomarkers(BIOMARKERS_FILE, lib_search, METADATA_FILE, sample_feature_table_df)
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
    st.info("Please provide the Task ID for the Library Search workflow and the Quant Tabel file or Task ID from which it can be retrieve, then click Run Analysis.")
