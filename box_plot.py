import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd
from typing import List


def create_food_boxplot(df, x_variable:str, y_variables:str,
                              filter_pattern:str, comparison_groups:List,
                              title:str):
    """Create overlay box plots with statistical comparisons for multiple variables in food analysis data. Each variable gets its own group of boxes side by side.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing the data
    x_variable : str,
        Column name for x-axis grouping variable
    y_variables : list,
        List of column names for y-axis measurement variables
    filter_pattern : str,
        Regex pattern to filter filename column
    comparison_groups : list,
        Groups to compare statistically
    title : str, default "Food Readout Analysis"
        Plot title

    Returns:
    --------
    tuple
        (plotly.graph_objects.Figure, str) - Figure object and SVG string
    """

    # Filter data based on filename pattern
    if 'filename' in df.columns:
        filtered_df = df[df[x_variable].str.contains(filter_pattern, na=False, regex=True)].copy()
    else:
        filtered_df = df.copy()

    # Ensure the required columns exist
    if x_variable not in filtered_df.columns:
        raise ValueError(f"Column '{x_variable}' not found in dataframe")

    missing_vars = [var for var in y_variables if var not in filtered_df.columns]
    if missing_vars:
        raise ValueError(f"Columns {missing_vars} not found in dataframe")

    # Get unique groups and colors
    groups = filtered_df[x_variable].unique()
    colors = px.colors.qualitative.Set1[:len(groups)]

    fig = go.Figure()

    # Calculate spacing for variables and groups
    n_vars = len(y_variables)
    n_groups = len(groups)
    group_width = 0.8  # Total width for each variable's group of boxes
    box_width = group_width / n_groups  # Width of individual boxes
    variable_spacing = 1.2  # Space between variable groups

    # Create x-positions and labels
    x_positions = []
    x_labels = []

    for var_idx, y_var in enumerate(y_variables):
        var_center = var_idx * variable_spacing
        for group_idx, group in enumerate(groups):
            # Position boxes within each variable group
            x_pos = var_center + (group_idx - (n_groups - 1) / 2) * box_width
            x_positions.append(x_pos)
        x_labels.append(y_var)

    # Add box plots
    pos_idx = 0
    legend_added = set()  # Track which groups have been added to legend

    for var_idx, y_var in enumerate(y_variables):
        for group_idx, group in enumerate(groups):
            group_data = filtered_df[filtered_df[x_variable] == group]

            # Prepare hover data - include filename if available
            if 'filename' in group_data.columns:
                customdata = group_data['filename'].values
                hovertemplate = (
                    f"<b>{group}</b><br>" +
                    f"Variable: {y_var}<br>" +
                    "Value: %{y}<br>" +
                    "Filename: %{customdata}<br>" +
                    "<extra></extra>"
                )
            else:
                customdata = None
                hovertemplate = (
                    f"<b>{group}</b><br>" +
                    f"Variable: {y_var}<br>" +
                    "Value: %{y}<br>" +
                    "<extra></extra>"
                )

            fig.add_trace(go.Box(
                y=group_data[y_var],
                x=[x_positions[pos_idx]] * len(group_data),
                name=group,
                boxpoints='all',
                jitter=0.3,
                pointpos=0,
                customdata=customdata,
                hovertemplate=hovertemplate,
                marker=dict(
                    color=colors[group_idx % len(colors)],
                    size=4,
                    opacity=0.6,
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                line=dict(color=colors[group_idx % len(colors)], width=2),
                fillcolor=colors[group_idx % len(colors)],
                opacity=0.7,
                width=box_width * 0.8,  # Make boxes slightly narrower
                showlegend=group not in legend_added,  # Only show each group once in legend
                legendgroup=group  # Group legend items by group name
            ))

            if group not in legend_added:
                legend_added.add(group)

            pos_idx += 1

        # Add statistical annotations for each variable
        _add_statistical_annotation_overlay(fig, filtered_df, x_variable, y_var,
                                            comparison_groups, groups, var_idx,
                                            variable_spacing, box_width)

    # Calculate x-axis tick positions (center of each variable group)
    x_tick_positions = [i * variable_spacing for i in range(n_vars)]

    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="black")
        ),
        xaxis=dict(
            title=dict(text="Variables", font=dict(size=14)),
            tickmode='array',
            tickvals=x_tick_positions,
            ticktext=y_variables,
            tickfont=dict(size=12),
            range=[-0.5, (n_vars - 1) * variable_spacing + 0.5]
        ),
        yaxis=dict(
            title=dict(text="Relative Intensity", font=dict(size=14)),
            tickfont=dict(size=12)
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif"),
        height=600,
        width=max(600, n_vars * 200 + 200),  # Dynamic width based on number of variables
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    return fig


def _add_statistical_annotation_overlay(fig, filtered_df, x_variable, y_variable,
                                        comparison_groups, groups, var_idx,
                                        variable_spacing, box_width):
    """Add statistical annotation for overlay plot"""

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
                p_text = "p < 0.01"
            elif p_value < 0.05:
                p_text = "p < 0.05"
            else:
                p_text = f"p = {p_value:.3f}"

            # Calculate annotation position for this variable
            var_data = filtered_df[y_variable]
            y_max = var_data.max()
            y_range = var_data.max() - var_data.min()
            annotation_y = y_max + 0.1 * y_range

            # Calculate x positions for the statistical annotation
            var_center = var_idx * variable_spacing
            n_groups = len(groups)

            # Find positions of comparison groups
            group1_idx = list(groups).index(comparison_groups[0])
            group2_idx = list(groups).index(comparison_groups[1])

            x1 = var_center + (group1_idx - (n_groups - 1) / 2) * box_width
            x2 = var_center + (group2_idx - (n_groups - 1) / 2) * box_width

            # Make sure x1 < x2 for the line
            if x1 > x2:
                x1, x2 = x2, x1

            # Add horizontal line for significance
            fig.add_shape(
                type="line",
                x0=x1, x1=x2,
                y0=annotation_y, y1=annotation_y,
                line=dict(color="black", width=1)
            )

            # Add vertical lines at ends
            fig.add_shape(
                type="line",
                x0=x1, x1=x1,
                y0=annotation_y, y1=annotation_y - 0.03 * y_range,
                line=dict(color="black", width=1)
            )

            fig.add_shape(
                type="line",
                x0=x2, x1=x2,
                y0=annotation_y, y1=annotation_y - 0.03 * y_range,
                line=dict(color="black", width=1)
            )

            # Add p-value text at the center
            fig.add_annotation(
                x=(x1 + x2) / 2,
                y=annotation_y + 0.02 * y_range,
                text=p_text,
                showarrow=False,
                font=dict(size=10, color="black")
            )


# Example usage
if __name__ == '__main__':
    def demonstrate_usage():
        """Demonstrate the multi-variable overlay box plot function"""

        # Create sample data with multiple variables
        np.random.seed(42)
        sample_data = {
            'filename': ['Vegan_sample_1.txt', 'Vegan_sample_2.txt', 'Vegan_sample_3.txt',
                         'Omni_sample_1.txt', 'Omni_sample_2.txt', 'Omni_sample_3.txt'] * 10,
            'Classifier': ['Vegan'] * 30 + ['Omnivore'] * 30,
            'Spinach': np.concatenate([
                np.random.normal(5, 1.5, 30),  # Vegan group
                np.random.normal(7, 1.8, 30)  # Omnivore group
            ]),
            'Tomato': np.concatenate([
                np.random.normal(8, 2, 30),  # Vegan group
                np.random.normal(6, 1.5, 30)  # Omnivore group
            ]),
            'Lettuce': np.concatenate([
                np.random.normal(6, 1.2, 30),  # Vegan group
                np.random.normal(5, 1.8, 30)  # Omnivore group
            ]),
            'Carrot': np.concatenate([
                np.random.normal(7, 1.8, 30),  # Vegan group
                np.random.normal(8, 2.2, 30)  # Omnivore group
            ])
        }

        df = pd.DataFrame(sample_data)

        # Example with multiple variables
        fig, svg = create_food_boxplot(
            df,
            x_variable='Classifier',
            y_variables=['Spinach', 'Tomato', 'Lettuce', 'Carrot'],
            title="Multi-Variable Food Analysis (Overlay)",
            comparison_groups=['Vegan', 'Omnivore']
        )

        return fig


    # Run demonstration
    fig = demonstrate_usage()

    # Uncomment to show plot
    # fig.show()