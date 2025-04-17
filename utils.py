import requests

import streamlit as st


@st.cache_data
def fetch_file(task_id: str, file_name: str) -> str:
    """
    Fetches a file from a given task ID and loads it into a pandas DataFrame.

    Args:
        task_id (str): The task ID to construct the file URL.
        file_name (str): The name of the file to fetch. Must be one of the predefined options.

    Returns:
        str: The path to the downloaded file.
    """
    # Define available options for file_name
    valid_file_names = ["merged_results.tsv", "clustering/tall_raw_data.tsv"]
    if file_name not in valid_file_names:
        raise ValueError(f"Invalid file name. Choose from: {valid_file_names}")

    input_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/{file_name}"
    response = requests.get(input_url)
    response.raise_for_status()  # Raise an error for failed requests

    output_file_path = f"output/{file_name.split('/')[-1]}"

    with open(output_file_path, 'w') as f:
        f.write(response.text)

    return output_file_path


if __name__ == '__main__':
    # Example usage
    task_id = "34e2b1b692444bf6ae37e71dd137c300"
    file_name = "merged_results.tsv"

    lib_search_df = fetch_file(task_id, file_name)
    print(lib_search_df)  # Display the first few rows of the DataFrame
