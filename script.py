import argparse
import os

import pandas as pd


def process_food_biomarkers(biomarkers_file, lib_search_file, metadata_file: str | pd.DataFrame | None,
                            quant_table: pd.DataFrame) -> dict:
    # Get file extensions and set separators
    biomarkers_ext = os.path.splitext(biomarkers_file)[1].lower()
    biomarkers_sep = '\t' if biomarkers_ext in ['.tsv', '.txt'] else ','

    # Load biomarkers file
    biomarkers_df = pd.read_csv(biomarkers_file, sep=biomarkers_sep)
    
    # Handle lib_search_file (can be DataFrame or file path)
    if isinstance(lib_search_file, pd.DataFrame):
        lib_search_df = lib_search_file
    else:
        lib_search_df = pd.read_csv(lib_search_file, sep='\t')
    
    # Use the provided quant_table DataFrame directly
    sample_feature_table_df = quant_table

    # Prepare library search data
    if "CompoundName" in lib_search_df.columns and lib_search_df["CompoundName"].astype(str).str.isdigit().all():
        lib_search_df = lib_search_df.rename(columns={
            "CompoundName": "Node",
        })[["#Scan#", "Node"]]
    else:
        lib_search_df = lib_search_df.rename(columns={
            "Protein": "Node"
        })[["#Scan#", "Node"]]
    lib_search_df = lib_search_df[["#Scan#", "Node"]].astype(str)
    lib_search_df['Node'] = lib_search_df['Node'].apply(lambda x: x.replace(':', ''))
    biomarkers_df = biomarkers_df.rename(columns={"Feature": "Node"})
    biomarker_hits = lib_search_df.astype(str).merge(biomarkers_df.astype(str), on="Node", how="left")

    # Transpose data
    sample_transpose = sample_feature_table_df.set_index("row ID")
    # sample_transpose = sample_transpose.filter(like=".mzML", axis=1).T.reset_index()
    sample_transpose = sample_transpose.filter(regex=r"\.mzML|\.mzXML", axis=1).T.reset_index()
    sample_transpose = sample_transpose.rename(columns={"index": "filename"})

    # Remove " Peak area" from filenames
    sample_transpose["filename"] = sample_transpose["filename"].str.replace(" Peak area", "", regex=False)
    sample_transpose = sample_transpose.sort_values("filename")

    # TIC normalization
    sample_tic = sample_transpose.set_index("filename")
    sample_tic = sample_tic + 1
    sample_tic = sample_tic.div(sample_tic.sum(axis=1), axis=0) * 1e5
    sample_tic = sample_tic.T.reset_index()
    sample_tic = sample_tic.rename(columns={"index": "row ID"})

    # Extract biomarkers from TIC-normalized data
    biomarker_scans = lib_search_df["#Scan#"].astype(str)
    biomarkers_in_samples = sample_tic[sample_tic["row ID"].astype(str).isin(biomarker_scans.values)]
    biomarkers_in_samples = biomarkers_in_samples.set_index("row ID").reset_index()
    biomarkers_in_samples = biomarkers_in_samples.rename(columns={"row ID": "#Scan#"})
    biomarkers_in_samples["#Scan#"] = biomarkers_in_samples["#Scan#"].astype(str)
    biomarker_hits['#Scan#'] = biomarker_hits['#Scan#'].astype(str)

    biomarkers_in_samples = biomarkers_in_samples.merge(biomarker_hits, on="#Scan#", how="left")
    biomarkers_in_samples = biomarkers_in_samples[biomarkers_in_samples["category"].notna()]

    # Separate multiple categories and summarize
    df_foods = biomarkers_in_samples.assign(category=biomarkers_in_samples["category"].str.split(", ")).explode(
        "category")
    food_summarized = df_foods.groupby("category").sum(numeric_only=True).reset_index()

    # Reshape for merging with metadata
    food_summarized = food_summarized.set_index("category").T.reset_index()
    food_summarized = food_summarized.rename(columns={"index": "filename"})
    food_summary_output = food_summarized.copy()
    food_summarized["filename"] = food_summarized["filename"].str.replace(" Peak area", "", regex=False)

    # Handle metadata file processing
    if isinstance(metadata_file, pd.DataFrame):
        # Metadata is already a DataFrame
        metadata_df = metadata_file
        metadata_df["filename"] = metadata_df["filename"].str.strip()
        food_summarized["filename"] = food_summarized["filename"].str.strip()
        food_metadata = food_summarized.merge(metadata_df, on="filename", how="left")
    elif isinstance(metadata_file, str) and metadata_file:
        # Metadata is a file path
        metadata_ext = os.path.splitext(metadata_file)[1].lower()
        metadata_sep = '\t' if metadata_ext in ['.tsv', '.txt'] else ','
        metadata_df = pd.read_csv(metadata_file, sep=metadata_sep)
        metadata_df["filename"] = metadata_df["filename"].str.strip()
        food_summarized["filename"] = food_summarized["filename"].str.strip()
        food_metadata = food_summarized.merge(metadata_df, on="filename", how="left")
    else:
        # No metadata provided
        food_metadata = None

    return {
        'result_df': food_metadata,
        'food_summary': food_summary_output
    }


if __name__ == '__main__':
    # --- Option 1: Run with command-line arguments ---
    # Uncomment below to use argparse for file-based inputs
    # parser = argparse.ArgumentParser(description="Process food biomarkers and generate metadata.")
    # parser.add_argument("--biomarkers_file", required=True, help="Path to the biomarkers CSV file.")
    # parser.add_argument("--lib_search_file", required=True, help="Path to the library search TSV file.")
    # parser.add_argument("--metadata_file", required=True, help="Path to the metadata TSV file.")
    # parser.add_argument("--sample_quant_table_file", required=True, help="Path to the sample feature table CSV file.")
    # args = parser.parse_args()
    # sample_quant_table_df = pd.read_csv(args.sample_quant_table_file, sep=None, engine='python')
    # lib_search_df = pd.read_csv(args.lib_search_file, sep='\t')
    # result = process_food_biomarkers(
    #     biomarkers_file=args.biomarkers_file,
    #     lib_search_file=lib_search_df,
    #     metadata_file=args.metadata_file,
    #     quant_table=sample_quant_table_df,
    # )

    # --- Option 2: Run with hardcoded variables (manual testing) ---
    # You can use either file paths or DataFrames for the inputs as supported by the function
    from gnpsdata import taskresult, workflow_fbmn

    def get_gnps2_df_wrapper(taskid, result_path):
        return taskresult.get_gnps2_task_resultfile_dataframe(taskid, result_path)

    def get_gnps2_fbmn_quant_table(taskid):
        return workflow_fbmn.get_quantification_dataframe(taskid, gnps2=True)

    # Example: using DataFrame for lib_search_file, file paths for others
    lib_search_df = get_gnps2_df_wrapper('a496eab9f98b41f7a571cf88b3d14977', "nf_output/merged_results.tsv")
    biomarker_file = './data/biomarker_tables/500_foods_level5.csv'
    metadata_file = '/path/to/file/metadata.tsv'
    quant_file = '/path/to/file/quantd_table.csv'
    sample_quant_table_df = pd.read_csv(quant_file, sep=None, engine='python')

    result = process_food_biomarkers(
        biomarkers_file=biomarker_file,
        lib_search_file=lib_search_df,  # DataFrame
        metadata_file=metadata_file,    # file path
        quant_table=sample_quant_table_df  # DataFrame
    )

    # Example: using only file paths (all inputs as strings)
    # result = process_food_biomarkers(
    #     biomarkers_file=biomarker_file,
    #     lib_search_file='nf_output/merged_results.tsv',
    #     metadata_file=metadata_file,
    #     quant_table=sample_quant_table_df
    # )

    # Print summary for verification
    print("Food metadata result (head):")
    print(result['result_df'].head() if result['result_df'] is not None else "No metadata result.")
    print("\nFood summary (head):")
    print(result['food_summary'].head())
