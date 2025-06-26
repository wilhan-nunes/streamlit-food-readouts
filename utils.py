import requests
import os
from typing import Literal

import streamlit as st


@st.cache_data
def fetch_file(
    task_id: str,
    file_name: str,
    type: Literal["quant_table", "library_search_table"],
    save_to_disk: bool = False,
    return_content: bool = True
) -> str | None | tuple[str, str]:
    """
    Fetches a file from a given task ID and optionally saves it to disk and/or returns its contents.

    Args:
        task_id (str): The task ID to construct the file URL.
        file_name (str): The name of the file to fetch. Must be one of the predefined options.
        type (Literal["quant_table", "library_search_table"]): The type of file to fetch.
        save_to_disk (bool): Whether to save the file to disk. Defaults to True.
        return_content (bool): Whether to return the file contents. Defaults to False.

    Returns:
        str: The path to the downloaded file if saved to disk and return_content is False.
        None: If not saving to disk and not returning content.
        tuple[str, str]: (output_file_path, content) if both saving to disk and returning content.
        str: The content if only returning content.
    """
    if type == "library_search_table":
        input_url = f"https://gnps2.org/resultfile?task={task_id}&file=nf_output/{file_name}"
    elif type == "quant_table":
        input_url = f"https://gnps2.org/result?task={task_id}&viewname=quantificationdownload&resultdisplay_type=task"
    response = requests.get(input_url)
    response.raise_for_status()
    content = response.content
    output_file_path = f"output/{file_name}"

    if save_to_disk:
        os.makedirs('output', exist_ok=True)
        with open(output_file_path, 'w') as f:
            f.write(content)
        if return_content:
            return output_file_path, content
        return output_file_path
    else:
        if return_content:
            return content
        return None

if __name__ == '__main__':
    # Example usage
    task_id = "34e2b1b692444bf6ae37e71dd137c300"
    file_name = "merged_results.tsv"

    lib_search_df = fetch_file(task_id, file_name)
    print(lib_search_df)  # Display the first few rows of the DataFrame
