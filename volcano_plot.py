import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.stats import ttest_ind

warnings.filterwarnings('ignore')


def create_interactive_volcano_plot(data,
                                    group_col: str,
                                    group1: str,
                                    group2: str,
                                    title: str,
                                    p_threshold=0.05,
                                    fc_threshold=0.5,
                                    exclude_cols=None,
                                    show_labels=True,
                                    top_n_labels=15,
                                    width=900,
                                    height=600):
    """
    Create an interactive volcano plot for metabolomics data comparison
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input dataframe containing metabolite data and group classifications
    group_col : str
        Column name containing group classifications
    group1 : str
        Reference group name (denominator in fold change calculation)
    group2 : str
        Comparison group name (numerator in fold change calculation)
    p_threshold : float
        P-value threshold for significance (default: 0.05)
    fc_threshold : float
        Log2 fold change threshold for significance (default: 0.5)
    exclude_cols : list
        List of column names to exclude from analysis (metadata columns)
    title : str
        Plot title
    show_labels : bool
        Whether to show labels for significant metabolites
    top_n_labels : int
        Number of top significant metabolites to label
    width : int
        Plot width in pixels
    height : int
        Plot height in pixels
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive volcano plot
    """

    # Default excluded columns if not provided
    if exclude_cols is None:
        exclude_cols = ['filename', 'Sample', 'description', 'Classifier', 'Sub_classifier']

    # Create a copy of the data to avoid modifying the original
    df = data.copy()

    # Identify metabolite columns (exclude metadata columns)
    metabolite_cols = [col for col in df.columns if col not in exclude_cols]

    # Filter data for the two groups of interest
    group1_data = df[df[group_col] == group1]
    group2_data = df[df[group_col] == group2]

    if len(group1_data) == 0 or len(group2_data) == 0:
        raise ValueError(f"One or both groups ({group1}, {group2}) not found in the data")

    # Calculate statistics for each metabolite
    results = []

    for metabolite in metabolite_cols:
        # Get values for each group, removing NaN values
        group1_vals = group1_data[metabolite].dropna()
        group2_vals = group2_data[metabolite].dropna()

        if len(group1_vals) < 2 or len(group2_vals) < 2:
            # Skip metabolites with insufficient data
            continue

        # Calculate means
        group1_mean = group1_vals.mean()
        group2_mean = group2_vals.mean()

        # Calculate log2 fold change (group2 vs group1)
        # Add small constant to avoid log(0) issues
        epsilon = 1e-6
        fold_change = np.log2((group2_mean + epsilon) / (group1_mean + epsilon))

        # Perform t-test
        try:
            t_stat, p_value = ttest_ind(group2_vals, group1_vals)
        except:
            p_value = 1.0  # Assign non-significant p-value if test fails

        # Calculate -log10(p-value)
        neg_log10_p = -np.log10(max(p_value, 1e-300))  # Prevent log(0)

        # Calculate correlation with group classification
        # Create binary encoding (0 = group1, 1 = group2)
        group_numeric = df[group_col].map({group1: 0, group2: 1})
        correlation = df[metabolite].corr(group_numeric)

        results.append({
            'Metabolite': metabolite,
            'FoldChange': fold_change,
            'PValue': p_value,
            'NegLog10P': neg_log10_p,
            'Correlation': correlation,
            'Group1_Mean': group1_mean,
            'Group2_Mean': group2_mean,
            'AbsFoldChange': abs(fold_change)
        })

    # Convert results to DataFrame
    volcano_data = pd.DataFrame(results)

    if volcano_data.empty:
        raise ValueError("No valid metabolites found for analysis")

    # Classify metabolites based on significance
    volcano_data['Significance'] = 'Not Significant'
    volcano_data.loc[
        (volcano_data['PValue'] < p_threshold) &
        (volcano_data['FoldChange'] > fc_threshold),
        'Significance'
    ] = f'Higher in {group2}'
    volcano_data.loc[
        (volcano_data['PValue'] < p_threshold) &
        (volcano_data['FoldChange'] < -fc_threshold),
        'Significance'
    ] = f'Higher in {group1}'

    # Create color mapping
    color_map = {
        f'Higher in {group2}': 'darkgreen',
        f'Higher in {group1}': 'darkred',
        'Not Significant': 'lightgray'
    }

    # Create the base scatter plot
    fig = go.Figure()

    # Add points for each significance category
    for significance, color in color_map.items():
        subset = volcano_data[volcano_data['Significance'] == significance]

        fig.add_trace(go.Scatter(
            x=subset['FoldChange'],
            y=subset['NegLog10P'],
            mode='markers',
            marker=dict(
                color=color,
                size=8,
                opacity=0.7,
                line=dict(width=1, color='white')
            ),
            name=significance,
            text=subset['Metabolite'],
            hovertemplate=(
                    '<b>%{text}</b><br>' +
                    f'log2(FC) {group2} vs {group1}: %{{x:.3f}}<br>' +
                    '-log10(p-value): %{y:.3f}<br>' +
                    f'{group1} mean: ' + subset['Group1_Mean'].round(3).astype(str) + '<br>' +
                    f'{group2} mean: ' + subset['Group2_Mean'].round(3).astype(str) + '<br>' +
                    'p-value: ' + subset['PValue'].map('{:.2e}'.format) +
                    '<extra></extra>'
            ),
            showlegend=True
        ))

    # Add significance threshold lines
    fig.add_hline(
        y=-np.log10(p_threshold),
        line_dash="dash",
        line_color="red",
        opacity=0.7,
        annotation_text=f"p = {p_threshold}",
        annotation_position="top right"
    )

    fig.add_vline(
        x=fc_threshold,
        line_dash="dash",
        line_color="blue",
        opacity=0.7
    )

    fig.add_vline(
        x=-fc_threshold,
        line_dash="dash",
        line_color="blue",
        opacity=0.7
    )

    # Add labels for top significant metabolites if requested
    if show_labels:
        # Select top significant metabolites based on p-value and fold change
        significant_metabolites = volcano_data[
            volcano_data['Significance'] != 'Not Significant'
            ].nlargest(top_n_labels, 'NegLog10P')

        for _, row in significant_metabolites.iterrows():
            fig.add_annotation(
                x=row['FoldChange'],
                y=row['NegLog10P'],
                text=row['Metabolite'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor="gray",
                ax=20,
                ay=-20,
                font=dict(size=10, color="black"),
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="gray",
                borderwidth=1
            )

    # Update layout
    fig.update_layout(
        title={
            'text': title,
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'family': 'Arial, sans-serif'}
        },
        xaxis_title=f'log2(Fold Change) - {group2} vs {group1}',
        yaxis_title='-log10(p-value)',
        width=width,
        height=height,
        template='plotly_white',
        hovermode='closest',
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01
        ),
        margin=dict(l=60, r=120, t=80, b=60)
    )

    # Update axes
    fig.update_xaxes(
        title_font_size=14,
        tickfont_size=12,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )

    fig.update_yaxes(
        title_font_size=14,
        tickfont_size=12,
        showgrid=True,
        gridwidth=1,
        gridcolor='lightgray'
    )

    return fig


# Example usage function
def example_usage():
    """
    Example of how to use the volcano plot function
    """
    # Assuming you have loaded your data as 'food_metadata'
    food_metadata = pd.read_csv('./output/food_metadata.csv')

    # Create the volcano plot
    fig = create_interactive_volcano_plot(
        data=food_metadata,
        group_col='Classifier',
        group1='Omnivore',
        group2='Vegan',
        p_threshold=0.05,
        fc_threshold=0.5,
        title="Interactive Volcano Plot: Vegan vs Omnivore Food Metabolites",
        show_labels=True,
        top_n_labels=15
    )

    # Display the plot
    fig.show()

    # Save the plot
    # fig.write_html("volcano_plot_interactive.html")
    # fig.write_image("volcano_plot.png", width=1200, height=800, scale=2)

    print("Example usage code provided in comments above")


# Run example
if __name__ == "__main__":
    example_usage()
