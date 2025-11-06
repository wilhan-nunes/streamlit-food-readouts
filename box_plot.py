import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import ttest_ind, kruskal, mannwhitneyu
import numpy as np
import pandas as pd
from typing import List


def create_food_boxplot(df, x_variable:str, y_variables:str,
                              filter_pattern:str, comparison_groups:List,
                              title:str, use_nonparametric:bool=True, 
                              log_scale:bool=False,
                              gridlines:bool=True):
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
    use_nonparametric : bool, default True
        If True, uses Mann-Whitney U (2 groups) or Kruskal-Wallis (>2 groups) test.
        If False, uses t-test for 2 groups.
    log_scale : bool, default False
        If True, uses logarithmic scale for y-axis. Useful for data spanning multiple orders of magnitude.

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

    # If using log scale, add small constant to avoid log(0)
    if log_scale:
        epsilon = 1e-9
        for var in y_variables:
            filtered_df[var] = filtered_df[var] + epsilon

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
    stats_results_dict = {}

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
        stats_result = _add_statistical_annotation_overlay(fig, filtered_df, x_variable, y_var,
                                            comparison_groups, groups, var_idx,
                                            variable_spacing, box_width, use_nonparametric)
        stats_results_dict[y_var] = stats_result

    # Calculate x-axis tick positions (center of each variable group)
    x_tick_positions = [i * variable_spacing for i in range(n_vars)]

    # Determine y-axis configuration
    y_axis_config = dict(
        title=dict(text="Relative Intensity" + (" (log scale)" if log_scale else ""), 
                   font=dict(size=14)),
        tickfont=dict(size=12)
    )
    
    if log_scale:
        y_axis_config['type'] = 'log'
    
    if not gridlines:
        y_axis_config['showgrid'] = False

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
        yaxis=y_axis_config,
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

    return fig, stats_results_dict


def _add_statistical_annotation_overlay(fig, filtered_df, x_variable, y_variable,
                                        comparison_groups, groups, var_idx,
                                        variable_spacing, box_width, test_type='auto'):
    """Add statistical annotation for overlay plot using relative positioning at the top of the figure
    
    Parameters:
    -----------
    test_type : str, default 'auto'
        Statistical test to use. Options: 'auto', 'kruskal', 'mannwhitney', 'ttest'
        - 'auto': Uses Mann-Whitney for 2 groups, Kruskal-Wallis for >2 groups
        - 'kruskal': Kruskal-Wallis H-test (non-parametric, for 2+ groups)
        - 'mannwhitney': Mann-Whitney U test (non-parametric, for 2 groups only)
        - 'ttest': Independent samples t-test (parametric, for 2 groups only)
    """

    if len(comparison_groups) < 2:
        return
    
    # Check if all comparison groups exist in the data
    if not all(group in groups for group in comparison_groups):
        return
    
    # Prepare data for all comparison groups
    group_data = []
    for group in comparison_groups:
        data = filtered_df[filtered_df[x_variable] == group][y_variable].dropna()
        if len(data) > 0:
            group_data.append(data)
        else:
            return  # Skip if any group has no data
    
    if len(group_data) < 2:
        return
    
    stat, p_value = None, None

    # Perform statistical test based on test_type
    if test_type == 'auto':
        if len(comparison_groups) == 2:
            stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
            test_name = "Mann-Whitney U"
        else:
            stat, p_value = kruskal(*group_data)
            test_name = "Kruskal-Wallis"
    elif test_type == 'kruskal':
        stat, p_value = kruskal(*group_data)
        test_name = "Kruskal-Wallis"
    elif test_type == 'mann_whitney':
        if len(comparison_groups) != 2:
            return {}  # Mann-Whitney requires exactly 2 groups
        stat, p_value = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
        test_name = "Mann-Whitney U"
    elif test_type == 'ttest':
        if len(comparison_groups) != 2:
            return {}  # t-test requires exactly 2 groups
        stat, p_value = ttest_ind(group_data[0], group_data[1])
        test_name = "t-test"
    else:
        raise ValueError(f"Invalid test_type: {test_type}. Choose from 'auto', 'kruskal', 'mannwhitney', 'ttest'")
    
    # Format p-value with test name
    if p_value < 0.001:
        p_text = f"{test_name}: p < 0.001***"
    elif p_value < 0.01:
        p_text = f"{test_name}: p < 0.01**"
    elif p_value < 0.05:
        p_text = f"{test_name}: p < 0.05*"
    else:
        p_text = f"{test_name}: p = {p_value:.3f} ns"
    
    # Calculate x positions for the statistical annotation
    var_center = var_idx * variable_spacing
    n_groups = len(groups)
    
    # Calculate x positions in data coordinates
    if len(comparison_groups) == 2:
        # Two-group comparison: center between the two groups
        group1_idx = list(groups).index(comparison_groups[0])
        group2_idx = list(groups).index(comparison_groups[1])
        
        x1 = var_center + (group1_idx - (n_groups - 1) / 2) * box_width
        x2 = var_center + (group2_idx - (n_groups - 1) / 2) * box_width
        
        # Make sure x1 < x2 for the line
        if x1 > x2:
            x1, x2 = x2, x1
        
        x_center = (x1 + x2) / 2
    else:
        # Multi-group comparison: center over all groups
        group_indices = [list(groups).index(group) for group in comparison_groups]
        x_positions = [var_center + (idx - (n_groups - 1) / 2) * box_width for idx in group_indices]
        
        x_center = sum(x_positions) / len(x_positions)
    
    # Add annotation using relative positioning (paper coordinates)
    # This places annotations at a fixed position relative to the figure top
    # y position is relative to paper (0-1 scale), starting from top
    annotation_y_paper = 1.05
    
    fig.add_annotation(
        x=x_center,
        y=annotation_y_paper,
        xref="x",  # x position in data coordinates
        yref="paper",  # y position relative to figure (0=bottom, 1=top)
        text=p_text,
        showarrow=False,
        font=dict(size=10, color="black"),
        bgcolor="rgba(255, 255, 255, 0.8)",  # Semi-transparent white background
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        xanchor="center",
        yanchor="top"
    )

    return {'statistic': stat, 'p_value': p_value}


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

        # Example with multiple variables using Kruskal-Wallis (default)
        fig = create_food_boxplot(
            df,
            x_variable='Classifier',
            y_variables=['Spinach', 'Tomato', 'Lettuce', 'Carrot'],
            filter_pattern='Vegan|Omni',
            title="Multi-Variable Food Analysis (Kruskal-Wallis)",
            comparison_groups=['Vegan', 'Omnivore'],
            use_nonparametric=True,  # Uses Mann-Whitney U for 2 groups
            log_scale=False  # Set to True to use log scale
        )

        return fig


    # Run demonstration
    fig = demonstrate_usage()

    # Uncomment to show plot
    # fig.show()