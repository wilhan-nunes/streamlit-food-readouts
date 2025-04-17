import requests
import os
from typing import Literal

import streamlit as st


@st.cache_data
def fetch_file(task_id: str, file_name: str, type: Literal["quant_table", "library_search_table"]) -> str:
    """
    Fetches a file from a given task ID and loads it into a pandas DataFrame.

    Args:
        task_id (str): The task ID to construct the file URL.
        file_name (str): The name of the file to fetch. Must be one of the predefined options.
        type (Literal["quant_table", "library_search_table"]): The type of file to fetch.

    Returns:
        str: The path to the downloaded file.
    """
    # Define available options for file_name
    if type == "library_search_table":
        input_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/{file_name}"
    elif type == "quant_table":
        input_url = f"https://gnps2.org/result?task={task_id}&viewname=quantificationdownload&resultdisplay_type=task"
    response = requests.get(input_url)
    response.raise_for_status()  # Raise an error for failed requests
    os.makedirs('output', exist_ok=True)
    output_file_path = f"output/{file_name}"

    with open(output_file_path, 'w') as f:
        f.write(response.text)

    return output_file_path


if __name__ == '__main__':
    # Example usage
    task_id = "34e2b1b692444bf6ae37e71dd137c300"
    file_name = "merged_results.tsv"

    lib_search_df = fetch_file(task_id, file_name)
    print(lib_search_df)  # Display the first few rows of the DataFrame
