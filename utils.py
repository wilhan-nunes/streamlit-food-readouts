from typing import List

import streamlit as st


def add_df_and_filtering(df, key_prefix: str, default_cols: List = None):
    # Session state for tracking number of filters
    if f"{key_prefix}_filter_count" not in st.session_state:
        st.session_state[f"{key_prefix}_filter_count"] = 0

    add_col, remove_col, _, _ = st.columns(4)
    with add_col:
        # Button to add more filter fields
        if st.button("➕ Add Filter Field", use_container_width=True, key=f"{key_prefix}_add_btn"):
            st.session_state[f"{key_prefix}_filter_count"] += 1
    with remove_col:
        if st.button("➖ Remove Filter Field", use_container_width=True, key=f"{key_prefix}_rmv_btn"):
            st.session_state[f"{key_prefix}_filter_count"] -= 1

    filtered_df = df.astype(str).copy()

    # Generate filter fields
    for i in range(st.session_state[f"{key_prefix}_filter_count"]):
        col1, col2 = st.columns([1, 2])
        with col1:
            if i == 0:
                st.markdown("**Filter Column**")
            selected_col = st.selectbox(
                f"Column {i + 1}", filtered_df.columns, key=f"{key_prefix}_col_select_{i}"
            )
        with col2:
            if i == 0:
                st.markdown("**Search String**")
            search_term = st.text_input(
                f"Contains (Column {i + 1})", key=f"{key_prefix}_search_input_{i}"
            )

        if selected_col and search_term:
            filtered_df = filtered_df[filtered_df[selected_col].str.contains(search_term, case=False, na=False)]

    # Show result
    st.markdown("### 🔎 Filtered Results")
    st.write(f"Total results: {len(filtered_df)}")
    all_cols = filtered_df.columns
    if default_cols:
        with st.expander('Cols to show'):
            cols_to_show = st.multiselect("Columns to show", options=all_cols, default=default_cols,
                                          label_visibility='collapsed')
    else:
        cols_to_show = all_cols

    return filtered_df[cols_to_show]
