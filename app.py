import os
from io import BytesIO

import pandas as pd
import streamlit as st

from pca_viz import create_pca_visualizations
from script import process_food_biomarkers
from utils import fetch_file
from box_plot import *
from volcano_plot import create_interactive_volcano_plot

# cute badges
BADGE_LIBRARY_ = ":green-badge[Library]"
BADGE_UPLOAD_QUANT_TABLE_ = ":blue-badge[Upload Quant Table]"
BADGE_QUANT_TABLE_ = ":orange-badge[Quant Table]"

BIOMARKERS_FOLDER = 'data/biomarker_tables'

# Streamlit app title
st.title("Food Biomarkers Analysis")

#defining query params to populate input fields
query_params = st.query_params
lib_task_id = query_params.get('lib_task_id', '')
quant_task_id = query_params.get('quant_task_id', '')

# Sidebar inputs
with st.sidebar:
    st.header("Inputs")

    files = os.listdir(BIOMARKERS_FOLDER)
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

    sample_quant_table_file = st.file_uploader(f"{BADGE_UPLOAD_QUANT_TABLE_} Upload Sample Feature Table",
                                               type=["csv", "tsv"])
    metadata_file = st.file_uploader('Upload Metadata File', type=["csv", "tsv"])

    if sample_quant_table_file:
        st.success(f"{BADGE_UPLOAD_QUANT_TABLE_}  will be used for analysis.")
    elif quant_table_task_id:
        st.success(f"No file uploaded. {BADGE_QUANT_TABLE_} task ID will be used to fetch the quant table.")
    else:
        st.warning(f"Please {BADGE_UPLOAD_QUANT_TABLE_} or provide a Task ID  for {BADGE_QUANT_TABLE_}")

    run_analysis = st.button("Run Analysis", help="Click to start the analysis with the provided inputs.",
                             use_container_width=True)

    reset_button = st.button("Reset Session", help="Click to clear the cache.", use_container_width=True, type="primary")
    if reset_button:
        st.session_state.clear()
        st.rerun()

# Process files when task ID and sample feature table are provided
if run_analysis:
    st.session_state['run_analysis'] = True
    try:
        # Retrieve lib_search using the task ID
        with st.spinner("Downloading library result table..."):
            lib_search = fetch_file(lib_search_task_id.strip(), "merged_results.tsv", type="library_search_table", save_to_disk=False,
                                    return_content=True)
            st.empty().success("Library result table downloaded successfully!")

        with st.spinner("Downloading FBMN Quant table from task ID..."):
            if sample_quant_table_file is None:
                sample_quant_table_file = fetch_file(quant_table_task_id.strip(), "quant_table.csv",
                                                     type="quant_table",
                                                     save_to_disk=False,
                                                     return_content=True)
                quant_table_contents = BytesIO(sample_quant_table_file)
                st.success(f"Quant table downloaded successfully from task {quant_table_task_id}!", icon="🔗")
            else:
                st.success("Sample Feature Table loaded from uploaded file successfully!", icon="📂")
                quant_table_contents = sample_quant_table_file

        # Load user-uploaded sample feature table
        sample_quant_table_df = pd.read_csv(quant_table_contents, sep=None, engine='python')

        # Process data
        with st.spinner("Processing data..."):
            biomarker_filepath = os.path.join(BIOMARKERS_FOLDER, selected_biomarkers_file)
            result = process_food_biomarkers(biomarker_filepath, lib_search, metadata_file, sample_quant_table_df)
            st.success("Data processed successfully!")

        # Load and display the resulting table
        st.session_state.result_file = result['result_file_path']
        st.session_state.result_dataframe = result['result_df']


    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise

if not st.session_state.get('run_analysis', False):
    st.info(
        ":information_source: Please, provide the inputs, then click Run Analysis.")

if "run_analysis" in st.session_state:

    result_data = st.session_state.get('result_dataframe', None)

    quantitative_cols = sorted(result_data.select_dtypes(include=['number']).columns.tolist())
    categorical_cols = sorted(result_data.select_dtypes(include=['object', 'category']).columns.tolist())
    st.session_state['quantitative_cols'] = quantitative_cols
    st.session_state['categorical_cols'] = categorical_cols

    st.subheader("Processed Food Metadata")
    st.expander("Table").dataframe(result_data)

    # Download option
    st.download_button(
        label="Download Processed Data",
        data=result_data.to_csv(sep='\t', index=False),
        file_name="food_metadata.tsv",
        mime="text/csv"
    )

    # Create box plot
    st.markdown("---")
    st.subheader("Box Plot of Food Biomarkers")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.selectbox('Select categorical variable for x-axis', categorical_cols, key='x_variable')
    with col_2:
        st.selectbox('Select numerical variable for y-axis', quantitative_cols, key='y_variable')

    box_plot_fig, box_plot_svg = create_food_boxplot(
        result_data,
        x_variable=st.session_state.get('x_variable', 'Classifier'),
        y_variable=st.session_state.get('y_variable', 'Spinach'),
        title=f"{st.session_state.y_variable} Readout Analysis",
           )
    st.plotly_chart(box_plot_fig, use_container_width=True)
    st.download_button(
        label="Download Box Plot SVG",
        data=box_plot_svg,
        file_name="food_box_plot.svg",
        mime="image/svg+xml"
    )

    # Create PCA visualizations
    st.markdown("---")
    st.subheader("PCA Visualizations")
    #input for PCA
    col1, col2 = st.columns(2)
    with col1:
        n_components = st.number_input("Number of PCA components", min_value=2, max_value=10, value=4, step=1)
        st.session_state.n_components = n_components
    with col2:
        # df is your DataFrame
        classifier_col = st.selectbox('Select Classifier Column', [None] + categorical_cols, key='classifier_col')

    results = create_pca_visualizations(
        result_data,
        classifier_col=classifier_col if classifier_col else 'Classifier',
        filter_patterns=['Omni', 'Vegan'],
        title_prefix="PCA NIST food readout",
        n_components= n_components,
        metadata_cols=['filename', 'Sample', 'description', 'Classifier', 'Sub_classifier']
    )

    plotly_figure = results['plotly_fig']
    mpl_figure = results['mpl_fig']
    svg_string = results['svg_string']
    pca_info = results['pca_results']

    col1, col2 = st.columns([2, 1])
    with col1:
        st.pyplot(mpl_figure, use_container_width=True)
    with col2:
        st.markdown("") #spacer
        variance_ratios = [pca_info['explained_variance_ratio'][i] for i in range(n_components)]
        st.markdown(
            f"Explained Variance Ratios: <br>"
            f"{', '.join([f'PC{i+1}={variance_ratios[i]:.2f}%' for i in range(n_components)])}",
            unsafe_allow_html=True
        )
        st.markdown(
            f"Explained variance: <br>"
            f"PC1={pca_info['explained_variance_ratio'][0]:.2f}%, PC2={pca_info['explained_variance_ratio'][1]:.2f}%<br>", unsafe_allow_html=True)
        st.markdown(f"Number of features used: {pca_info['n_features']}")
        st.expander('Feature Names', expanded=False).markdown(
            f"{', '.join(pca_info['feature_names'])}...", unsafe_allow_html=False)
    # st.image(svg_string)

    ## Volcano plot
    st.markdown("---")
    st.subheader("Volcano Plot")

    col1, col2, col3 = st.columns(3)

    with (col1):
        group_col = st.selectbox('Group Column', [None] + categorical_cols, key='group_col')
        if group_col:
            subcol1, subcol2 = st.columns(2)
            available_groups = result_data[group_col].unique().tolist()
            with subcol1:
                group1 = st.selectbox('Group 1', available_groups, index=0)
            with subcol2:
                group2 = st.selectbox('Group 2', available_groups, index=1)

    with col2:
        p_threshold = st.number_input('P-value threshold', value=0.05, min_value=0.001, max_value=1.0, step=0.01)
        fc_threshold = st.number_input('Fold change threshold', value=0.5, min_value=0.1, max_value=5.0, step=0.1)

    with col3:
        top_n_labels = st.number_input('Top N labels', value=15, min_value=0, max_value=50, step=1)
        show_labels = st.checkbox('Show labels', value=True)


    if group_col and group1 and group2:
        fig = create_interactive_volcano_plot(
            data=result_data,
            group_col=group_col,
            group1=group1,
            group2=group2,
            p_threshold=p_threshold,
            fc_threshold=fc_threshold,
            title=f"Interactive Volcano Plot: {group1} vs {group2}",
            show_labels=show_labels,
            top_n_labels=int(top_n_labels)
        )

        st.plotly_chart(fig, use_container_width=True)

else:
    with open('sop_food_readout.md', 'r') as f:
        sop_content = f.read()
    st.markdown(sop_content, unsafe_allow_html=True)