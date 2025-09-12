import os
from gnpsdata import workflow_fbmn, taskresult
import streamlit as st
from streamlit.components.v1 import html

from pca_viz import create_pca_visualizations
from script import process_food_biomarkers
from box_plot import *
from utils import add_df_and_filtering
from volcano_plot import create_interactive_volcano_plot
from ontology import load_ontology, add_sankey_ontology
import re


def get_git_short_rev():
    try:
        with open('.git/logs/HEAD', 'r') as f:
            last_line = f.readlines()[-1]
            hash_val = last_line.split()[1]
        return hash_val[:7]
    except Exception:
        return ".git/ not found"



# TODO: Bump version
app_version = "2025-08-15"
git_hash = get_git_short_rev()
repo_link = "https://github.com/wilhan-nunes/streamlit-food-readouts/"


# Streamlit app config and title
st.set_page_config(
    page_title="Dietary Intake Analysis - MetaboApp",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": (f"**App version**: {app_version} | "
                          f"[**Git Hash**: {git_hash}]({repo_link}/commit/{git_hash})")},
)

# Add a tracking token
html('<script async defer data-website-id="<your_website_id>" src="https://analytics.gnps2.org/umami.js"></script>', width=0, height=0)

# cute badges
BADGE_LIBRARY_ = ":green-badge[Library]"
BADGE_UPLOAD_QUANT_TABLE_ = ":blue-badge[Upload Quant Table]"
BADGE_QUANT_TABLE_ = ":orange-badge[Quant Table]"

BIOMARKERS_FOLDER = 'data/biomarker_tables'

#TODO populate other examples
EXAMPLES_CONFIG = {
    "load_example_nist": {'button_label': "NIST Vegan/Omnivore dataset",
                          'library_results': 'examples/01_NIST_library_search_results.tsv',
                          'quant_table': 'examples/01_NIST_Vegan_Ominvore_iimn_gnps_quant.csv',
                          'metadata_file': 'examples/01_NIST_Vegan_Omnivore_metadata.csv'},
    # "load_example_2": {'button_label': "Example 2: ...",
    #                    'library_results': 'examples/02_example2.tsv',
    #                    'quant_table': 'examples/02_example2.csv',
    #                    'metadata_file': 'examples/02_example2.csv'},
    # "load_example_3": {'button_label': "Example 3: ...",
    #                    'library_results': 'examples/03_example3.tsv',
    #                    'quant_table': 'examples/03_example3.csv',
    #                    'metadata_file': 'examples/03_example3.csv'},
}

st.title("🥗 Estimation of Dietary intake from untargeted metabolomics data")

# defining query params to populate input fields
query_params = st.query_params
lib_task_id = query_params.get('lib_task_id', '')
quant_task_id = query_params.get('quant_task_id', '')

# Sidebar inputs
with st.sidebar:
    st.image("foodreadout_logo.svg", width=200)
    st.header("Inputs", help='Provide the necessary inputs to run the analysis or select an example below.')
    with st.expander("Example Datasets", expanded=False):
        for name in EXAMPLES_CONFIG:
            button_label = EXAMPLES_CONFIG.get(name)['button_label']
            button_key = name

            st.button(button_label, type='tertiary', key=button_key,
                      help='Please also select the biomarkers file below to run with this example.')

    files = os.listdir(BIOMARKERS_FOLDER)
    BIOMARKERS_FILES = [f for f in files if f.endswith(('.csv', '.tsv'))]
    BIOMARKERS_FILES_DISPLAY = {
        f.split("_")[-1].replace('.csv', ''): f
        for f in BIOMARKERS_FILES if "level" in f
    }
    for f in BIOMARKERS_FILES:
        if f not in BIOMARKERS_FILES_DISPLAY.values():
            BIOMARKERS_FILES_DISPLAY[f.split('.')[0]] = f

    selected_biomarkers_file = st.selectbox(
        "Select Biomarkers File",
        BIOMARKERS_FILES_DISPLAY.keys(),
        help="Choose the biomarkers file to use for the analysis."
    )
    biomarker_filepath = os.path.join(BIOMARKERS_FOLDER, BIOMARKERS_FILES_DISPLAY.get(selected_biomarkers_file))

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
    sample_quant_table_file = None
    if not quant_table_task_id:
        sample_quant_table_file = st.file_uploader(f"{BADGE_UPLOAD_QUANT_TABLE_} Upload Sample Feature Table",
                                                   type=["csv", "tsv"])
        if sample_quant_table_file:
            st.success(f"{BADGE_UPLOAD_QUANT_TABLE_}  will be used for analysis.")
        elif quant_table_task_id:
            st.success(f"No file uploaded. {BADGE_QUANT_TABLE_} task ID will be used to fetch the quant table.")
        else:
            st.warning(f"Please {BADGE_UPLOAD_QUANT_TABLE_} or provide a Task ID  for {BADGE_QUANT_TABLE_}")

    metadata_file = st.file_uploader('Upload Metadata File', type=["csv", "tsv"])
    if not metadata_file:
        st.warning("Please upload the metadata file to continue", icon=":material/arrow_warm_up:")

    example_run = [name for name in st.session_state.keys() if
                   name.startswith('load_example_') and st.session_state[name]]

    # Store metadata availability in session state
    if metadata_file is not None:
        st.session_state['has_metadata'] = True
    elif example_run:  # If running example, metadata is available
        st.session_state['has_metadata'] = True

    run_analysis = st.button("Run Analysis", help="Click to start the analysis with the provided inputs.",
                             use_container_width=True, disabled=not(lib_search_task_id and (quant_table_task_id or sample_quant_table_file)), icon=":material/play_arrow:")

    reset_button = st.button("Reset Session", help="Click to clear the cache.", use_container_width=True,
                             type="primary", icon=":material/replay:")
    if reset_button:
        st.session_state.clear()
        if 'has_metadata' in st.session_state:
            del st.session_state['has_metadata']
        st.rerun()

    st.subheader("Contributors")
    st.markdown(
        """
    - [Harsha Gouda PhD](https://scholar.google.com/citations?user=mP5z-HsAAAAJ) - UC San Diego
    - [Wilhan Nunes PhD](https://scholar.google.com/citations?user=4cPVoeIAAAAJ) - UC San Diego
    - [Mingxun Wang PhD](https://www.cs.ucr.edu/~mingxunw/) - UC Riverside
    """
    )

# Process files when task ID and sample feature table are provided
# Check if any example button was pressed
example_run = [name for name in st.session_state.keys() if name.startswith('load_example_') and st.session_state[name]]

@st.cache_data
def get_gnps2_df_wrapper(taskid, result_path):
    return taskresult.get_gnps2_task_resultfile_dataframe(taskid, result_path)


@st.cache_data
def get_gnps2_fbmn_quant_table(taskid):
    return workflow_fbmn.get_quantification_dataframe(taskid, gnps2=True)


if run_analysis or example_run:
    st.session_state['run_analysis'] = True
    try:
        placeholder = st.empty()
        if run_analysis:
            # Retrieve lib_search using the task ID
            with st.spinner("Downloading library result table..."):
                lib_search = get_gnps2_df_wrapper(lib_search_task_id, "nf_output/merged_results.tsv")
                placeholder.success("Library result table downloaded successfully!")

            with st.spinner("Downloading FBMN Quant table from task ID..."):
                if sample_quant_table_file is None:
                    sample_quant_table_df = get_gnps2_fbmn_quant_table(quant_table_task_id)
                    placeholder.success(f"Quant table downloaded successfully from task {quant_table_task_id}!", icon="🔗")
                else:
                    st.success("Sample Feature Table loaded from uploaded file successfully!", icon="📂")
                    quant_table_contents = sample_quant_table_file
                    # Load user-uploaded sample feature table
                    sample_quant_table_df = pd.read_csv(quant_table_contents, sep=None, engine='python')

        else:
            example_files_dict = EXAMPLES_CONFIG.get(example_run[0], None)
            # Load example data from predefined files for demo mode
            with st.spinner("Loading example files..."):
                lib_search = pd.read_csv(example_files_dict.get('library_results'), sep='\t')
                sample_quant_table_df = pd.read_csv(example_files_dict.get('quant_table'))
                metadata_file = example_files_dict.get('metadata_file')
                st.toast("Example files loaded successfully!", icon="📂")

        with st.spinner("Processing data..."):
            result = process_food_biomarkers(biomarker_filepath, lib_search, metadata_file, sample_quant_table_df)
            st.toast("Data processed successfully!", icon="✅")
            placeholder.empty()
            # st.session_state.result_file = result['result_file_path']
            st.session_state.result_dataframe = result['result_df']
            st.session_state.food_summary = result['food_summary']



    except Exception as e:
        st.error(f"An error occurred: {e}")
        raise

if st.session_state.get('run_analysis', False) and not st.session_state.get('has_metadata', False):
    st.subheader("Sample Food Annotation Summary")
    st.info("This is a limited result based on the provided inputs. If you want to run the full analysis, please upload a metadata file with the sample feature table.")
    filtered_df = add_df_and_filtering(st.session_state.get('food_summary', pd.DataFrame()), key_prefix='food_summary')
    st.dataframe(filtered_df, use_container_width=True)
    st.download_button(
        label="Download Sample Food Summary",
        data=st.session_state.get('food_summary', pd.DataFrame()).to_csv(sep='\t', index=False),
        icon=":material/download:",
        file_name="food_summary.tsv",
        mime="text/csv"
    )

elif st.session_state.get('run_analysis', False) and st.session_state.get('has_metadata', False):
    result_data = st.session_state.get('result_dataframe', None)

    quantitative_cols = sorted(result_data.select_dtypes(include=['number']).columns.tolist())
    categorical_cols = sorted(result_data.select_dtypes(include=['object', 'category']).columns.tolist())
    st.session_state['quantitative_cols'] = quantitative_cols
    st.session_state['categorical_cols'] = categorical_cols

    st.subheader("Processed Food Metadata Table")
    with st.expander("Click to expand", expanded=False):
        filtered_with_metadata = add_df_and_filtering(result_data, key_prefix='food_metadata')
        st.dataframe(filtered_with_metadata)

    # Download option
    st.download_button(
        label="Download Processed Data",
        data=result_data.to_csv(sep='\t', index=False),
        icon=":material/download:",
        file_name="food_metadata.tsv",
        mime="text/csv"
    )

    # Load ontology
    #TODO: create new ontology for UK food files
    st.subheader("🌱 Sample Food Ontology")
    ontology = load_ontology('food_ontology.yaml')
    level_match = re.search(r'level(\d+)', selected_biomarkers_file)
    if level_match:
        level = level_match.groups(1)[0]
        fig = add_sankey_ontology(ontology, result_data, target_level=int(level), peak_threshold=0)
        height_by_level = {
            3: 800,
            4: 1000,
            5: 1500
        }
        fig_height = st.slider("Plot height", min_value=500, max_value=1500, value=height_by_level.get(int(level)), key='sankey_height')
        fig.update_layout(height=fig_height)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sorry! No ontology information for the selected biomarkers file at the moment.")

    # Create box plot
    st.markdown("---")
    st.subheader("Box Plot of Food Biomarkers")

    col_1, col_2 = st.columns(2)
    with col_1:
        st.selectbox('Select categorical variable for x-axis', categorical_cols, key='x_variable', index=0)
        if 'x_variable' in st.session_state:
            available = result_data[st.session_state.x_variable].unique().tolist()
            selected_groups = st.multiselect("Select the groups to compare", available, default=available[:2],)
    with col_2:
        st.multiselect('Select numerical variable for y-axis', quantitative_cols, key='y_variables', default=quantitative_cols[0])

    box_plot_fig, box_plot_svg = create_food_boxplot(
        result_data,
        x_variable=st.session_state.get('x_variable', 'Classifier'),
        y_variables=st.session_state.get('y_variables', ['Spinach']),
        filter_pattern="|".join(selected_groups),
        comparison_groups=selected_groups,
        title=f"{",".join(st.session_state.y_variables)} Readout Analysis",
    )
    st.plotly_chart(box_plot_fig, use_container_width=True)
    st.download_button(
        label="Download Box Plot SVG",
        data=box_plot_svg,
        icon=":material/download:",
        file_name="food_box_plot.svg",
        mime="image/svg+xml"
    )

    # Create PCA visualizations
    st.markdown("---")
    st.subheader("PCA Visualizations")
    # input for PCA
    col1, col2 = st.columns(2)
    with col1:
        n_components = st.number_input("Number of PCA components", min_value=2, max_value=10, value=2, step=1)
        st.session_state.n_components = n_components
    with col2:
        # df is your DataFrame
        classifier_col = st.selectbox('Select Classifier Column', [None] + categorical_cols, key='classifier_col')
        groups_to_compare = st.multiselect(
            'Select Groups to Compare',
            result_data[classifier_col].unique().tolist() if classifier_col else [],
            default=result_data[classifier_col].unique().tolist()[:2] if classifier_col else [],
            key='groups_to_compare'
        )
    if classifier_col is None:
        st.info("Select a Classifier Column above")
    else:
        results = create_pca_visualizations(
            result_data,
            classifier_col=classifier_col if classifier_col else 'Classifier',
            filter_patterns=groups_to_compare,
            title_prefix="PCA",
            n_components=n_components,
            metadata_cols=categorical_cols,
        )

        plotly_figure = results['plotly_fig']
        mpl_figure = results['mpl_fig']
        svg_string = results['svg_string']
        pca_info = results['pca_results']


        col1, col2 = st.columns([2, 1])
        with col1:
            st.pyplot(mpl_figure, use_container_width=True)
            # add svg download button
            st.download_button(
                label="Download PCA Plot SVG",
                data=svg_string,
                icon=":material/download:",
                file_name="pca_plot.svg",
                mime="image/svg+xml"
            )
        with col2:
            st.markdown("")  # spacer
            variance_ratios = [pca_info['explained_variance_ratio'][i] for i in range(n_components)]
            st.markdown(
                f"Explained Variance Ratios: <br>"
                f"{', '.join([f'PC{i + 1}={variance_ratios[i]:.2f}%' for i in range(n_components)])}",
                unsafe_allow_html=True
            )
            st.markdown(
                f"Explained variance: <br>"
                f"PC1={pca_info['explained_variance_ratio'][0]:.2f}%, PC2={pca_info['explained_variance_ratio'][1]:.2f}%<br>",
                unsafe_allow_html=True)
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
            top_n_labels=int(top_n_labels),
            exclude_cols=categorical_cols,
        )

        st.plotly_chart(fig, use_container_width=True)
        # add volcano plot svg download
        volcano_svg = fig.to_image(format="svg", width=800, height=600, scale=2)
        st.download_button(
            label="Download Volcano Plot SVG",
            data=volcano_svg,
            icon=":material/download:",
            file_name="volcano_plot.svg",
            mime="image/svg+xml"
        )

else:
    with open('sop_food_readout.md', 'r') as f:
        sop_content = f.read()
    st.markdown(sop_content, unsafe_allow_html=True)
    st.info("""
    - This application is part of the GNPS downstream analysis ecosystem known as **MetaboApps**.
    - If you encounter any issues or have suggestions, please reach out to the app maintainers.
    - [Checkout other tools](https://wang-bioinformatics-lab.github.io/GNPS2_Documentation/toolindex/#gnps2-web-tools)
    """)
