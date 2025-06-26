import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind


def create_food_boxplot(df, x_variable='classifier', y_variable='Spinach',
                        filter_pattern=r'Vegan|Omni', comparison_groups=['Vegan', 'Omnivore'],
                        title="Food Readout Analysis"):
    """
    Create a box plot with statistical comparisons for food analysis data.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    x_variable : str, default 'Classifier'
        Column name for x-axis grouping variable
    y_variable : str, default 'Spinach'
        Column name for y-axis measurement variable
    filter_pattern : str, default r'Vegan|Omni'
        Regex pattern to filter filename column
    comparison_groups : list, default ['Vegan', 'Omnivore']
        Groups to compare statistically
    title : str, default "Food Readout Analysis"
        Plot title
    return_svg : bool, default True
        Whether to return SVG string or plotly figure object

    Returns:
    --------
    str or plotly.graph_objects.Figure
        SVG string if return_svg=True, otherwise plotly figure
    """

    # Filter data based on filename pattern
    if 'filename' in df.columns:
        filtered_df = df[df['filename'].str.contains(filter_pattern, na=False, regex=True)].copy()
    else:
        filtered_df = df.copy()

    # Ensure the required columns exist
    if x_variable not in filtered_df.columns:
        raise ValueError(f"Column '{x_variable}' not found in dataframe")
    if y_variable not in filtered_df.columns:
        raise ValueError(f"Column '{y_variable}' not found in dataframe")

    # Create the box plot
    fig = go.Figure()

    # Get unique groups
    groups = filtered_df[x_variable].unique()
    colors = px.colors.qualitative.Set1[:len(groups)]

    # Add box plots for each group
    for i, group in enumerate(groups):
        group_data = filtered_df[filtered_df[x_variable] == group]

        fig.add_trace(go.Box(
            y=group_data[y_variable],
            x=[group] * len(group_data),
            name=group,
            boxpoints='all',
            jitter=0.3,
            pointpos=0,
            marker=dict(
                color=colors[i % len(colors)],
                size=6,
                opacity=0.6,
                line=dict(width=2, color='DarkSlateGrey')
            ),
            line=dict(color=colors[i % len(colors)], width=2),
            fillcolor=colors[i % len(colors)],
            opacity=0.7
        ))

    # Perform statistical test if comparison groups are specified
    p_value = None
    if len(comparison_groups) == 2 and all(group in groups for group in comparison_groups):
        group1_data = filtered_df[filtered_df[x_variable] == comparison_groups[0]][y_variable].dropna()
        group2_data = filtered_df[filtered_df[x_variable] == comparison_groups[1]][y_variable].dropna()

        if len(group1_data) > 0 and len(group2_data) > 0:
            # Perform t-test
            t_stat, p_value = ttest_ind(group1_data, group2_data)

            # Format p-value
            if p_value < 0.001:
                p_text = "p < 0.001"
            elif p_value < 0.01:
                p_text = f"p < 0.01"
            elif p_value < 0.05:
                p_text = f"p < 0.05"
            else:
                p_text = f"p = {p_value:.3f}"

            # Add significance annotation
            y_max = filtered_df[y_variable].max()
            y_range = filtered_df[y_variable].max() - filtered_df[y_variable].min()
            annotation_y = y_max + 0.1 * y_range

            # Add horizontal line for significance
            fig.add_shape(
                type="line",
                x0=0, x1=1,
                y0=annotation_y, y1=annotation_y,
                line=dict(color="black", width=1)
            )

            # Add vertical lines at ends
            fig.add_shape(
                type="line",
                x0=0, x1=0,
                y0=annotation_y, y1=annotation_y - 0.03 * y_range,
                line=dict(color="black", width=1)
            )

            fig.add_shape(
                type="line",
                x0=1, x1=1,
                y0=annotation_y, y1=annotation_y - 0.03 * y_range,
                line=dict(color="black", width=1)
            )

            # Add p-value text
            fig.add_annotation(
                x=0.5,
                y=annotation_y + 0.02 * y_range,
                text=p_text,
                showarrow=False,
                font=dict(size=12, color="black")
            )

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color="black")
        ),
        xaxis=dict(
            title=dict(text="Group", font=dict(size=14)),
            tickfont=dict(size=12),
            # linecolor="black",
            # linewidth=1.5
        ),
        yaxis=dict(
            title=dict(text="Relative Intensity", font=dict(size=14)),
            tickfont=dict(size=12),
            # linecolor="black",
            # linewidth=1.5
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif"),
        showlegend=False,
        width=550,
        height=600,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    # Add border around plot
    # fig.update_layout(
    #     shapes=[
    #         dict(
    #             type="rect",
    #             xref="paper", yref="paper",
    #             x0=0, y0=0, x1=1, y1=1,
    #             line=dict(color="black", width=2),
    #             fillcolor="rgba(0,0,0,0)"
    #         )
    #     ]
    # )

    # Return SVG string or figure object
    return fig, fig.to_image(format="svg", engine="kaleido").decode('utf-8')

# # Example usage function
# def demonstrate_usage():
#     """
#     Demonstrate how to use the create_food_boxplot function
#     """
#     # Create sample data
#     np.random.seed(42)
#     sample_data = {
#         'filename': ['Vegan_sample_1.txt', 'Vegan_sample_2.txt', 'Vegan_sample_3.txt',
#                      'Omni_sample_1.txt', 'Omni_sample_2.txt', 'Omni_sample_3.txt'] * 10,
#         'Classifier': ['Vegan'] * 30 + ['Omnivore'] * 30,
#         'Spinach': np.concatenate([
#             np.random.normal(5, 1.5, 30),  # Vegan group
#             np.random.normal(7, 1.8, 30)  # Omnivore group
#         ]),
#         'Tomato': np.concatenate([
#             np.random.normal(8, 2, 30),  # Vegan group
#             np.random.normal(6, 1.5, 30)  # Omnivore group
#         ])
#     }
#
#     df = pd.DataFrame(sample_data)
#
#     # Create box plot
#     svg_result = create_food_boxplot(
#         df,
#         x_variable='Classifier',
#         y_variable='Spinach',
#         title="Spinach Readout Analysis",
#         return_svg=False,
#     )
#
#     print("Box plot created successfully!")
#     print("Function parameters:")
#     print("- df: Your dataframe")
#     print("- x_variable: Column for grouping (default: 'Classifier')")
#     print("- y_variable: Column for measurements (default: 'Spinach')")
#     print("- filter_pattern: Regex for filename filtering (default: r'Vegan|Omni')")
#     print("- comparison_groups: Groups to compare (default: ['Vegan', 'Omnivore'])")
#     print("- title: Plot title")
#     print("- return_svg: Return SVG string (True) or plotly figure (False)")
#
#     return svg_result
#
# # Uncomment the line below to see a demonstration
# # demonstrate_usage()
