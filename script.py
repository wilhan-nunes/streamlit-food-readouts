import argparse
import os
from io import BytesIO

import pandas as pd


def process_food_biomarkers(biomarkers_file, lib_search_file, metadata_file: str | None, quant_table: pd.DataFrame,
                            output_dir="./output") -> dict:
    # Get file extensions and set separators
    biomarkers_ext = os.path.splitext(biomarkers_file)[1].lower()

    biomarkers_sep = '\t' if biomarkers_ext in ['.tsv', '.txt'] else ','

    if metadata_file:
        if isinstance(metadata_file, str):
            metadata_ext = os.path.splitext(metadata_file)[1].lower()
            metadata_sep = '\t' if metadata_ext in ['.tsv', '.txt'] else ','
        else:
            metadata_sep = None

        metadata_df = pd.read_csv(metadata_file, sep=metadata_sep)


    # Load files
    biomarkers_df = pd.read_csv(biomarkers_file, sep=biomarkers_sep)
    # lib_search_df = pd.read_csv(BytesIO(lib_search_file), sep='\t')
    lib_search_df = lib_search_file
    sample_feature_table_df = quant_table

    # Prepare library search data
    lib_search_df = lib_search_df.rename(columns={"Protein": "Node"})[["#Scan#", "Node"]]
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

    if metadata_file:
        metadata_df["filename"] = metadata_df["filename"].str.strip()
        food_summarized["filename"] = food_summarized["filename"].str.strip()

        food_metadata = food_summarized.merge(metadata_df, on="filename", how="left")

        # Save output
        # os.makedirs(output_dir, exist_ok=True)
        # result_path = f"{output_dir}/food_metadata.csv"
        # food_metadata.to_csv(result_path, index=False)
    else:
        result_path = None
        food_metadata = None

    return {
        'result_file_path': result_path,
        'result_df': food_metadata,
        'food_summary': food_summary_output
    }




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process food biomarkers and generate metadata.")
    parser.add_argument("--biomarkers_file", required=True, help="Path to the biomarkers CSV file.")
    parser.add_argument("--lib_search_file", required=True, help="Path to the library search TSV file.")
    parser.add_argument("--metadata_file", required=True, help="Path to the metadata TSV file.")
    parser.add_argument("--sample_quant_table_file", required=True, help="Path to the sample feature table CSV file.")
    parser.add_argument("--output_dir", default="./output", help="Directory to save the output file.")

    args = parser.parse_args()

    sample_feature_table_df = pd.read_csv(args.sample_feature_table_file, sep=None, engine='python')

    result = process_food_biomarkers(
        biomarkers_file=args.biomarkers_file,
        lib_search_file=args.lib_search_file,
        metadata_file=args.metadata_file,
        quant_table=sample_feature_table_df,
        output_dir=args.output_dir
    )

    output_file = result['result_file_path']
    print(f"Processed food metadata saved to: {output_file}")

# running command example:
# python script.py --biomarkers_file data/Biomarkers_level5_FC5_VIP6.csv --lib_search_file input_test_files/merged_results.tsv --metadata_file data/gnps_metadata_ming.tsv --sample_quant_table_file input_test_files/MSV82493_iimn_gnps_quant.csv


# Another example coming from task IDs:
# --biomarkers_file
# data/Biomarkers_level5_FC5_VIP6.csv
# --lib_search_file
# output/merged_results.tsv
# --metadata_file
# data/gnps_metadata_ming.tsv
# --sample_quant_table_file
# output/tall_raw_data.tsv
