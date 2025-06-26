import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
import io
import base64
from scipy import stats

def create_pca_visualizations(data, classifier_col='Classifier',
                            filter_patterns=['Omni', 'Vegan'], 
                            title_prefix="PCA NIST food readout",
                            n_components=4,
                            metadata_cols=['Sample', 'description', 'Classifier', 'Sub_classifier']):
    """
    Create PCA visualizations returning both Plotly figure and SVG string.
    
    Parameters:
    -----------
    data : pd.DataFrame
        The complete dataset with features and metadata columns
    classifier_col : str
        Column name containing group classifications
    filter_patterns : list
        Patterns to filter data based on classifier column (e.g., ['Omni', 'Vegan'])
    title_prefix : str
        Title prefix for the plots
    n_components : int
        Number of PCA components to compute
    metadata_cols : list
        List of column names that are metadata (not features)

    Returns:
    --------
    dict : Contains 'plotly_fig', 'svg_string', and 'pca_results'
    """
    
    # Separate features from metadata
    feature_cols = [col for col in data.columns if col not in metadata_cols]
    features = data[feature_cols]
    metadata = data[metadata_cols]

    # Filter data based on patterns in classifier column
    if filter_patterns:
        pattern = '|'.join(filter_patterns)
        mask = data[classifier_col].str.contains(pattern, na=False)
        data_filtered = data[mask].copy()
        features_filtered = features[mask].copy()
        metadata_filtered = metadata[mask].copy()
    else:
        data_filtered = data.copy()
        features_filtered = features.copy()
        metadata_filtered = metadata.copy()

    # Perform PCA on features only
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_filtered)
    
    pca = PCA(n_components=n_components)
    pca_scores = pca.fit_transform(features_scaled)
    
    # Create DataFrame with PCA scores and metadata
    pca_df = pd.DataFrame(pca_scores, 
                         columns=[f'PC{i+1}' for i in range(n_components)],
                         index=features_filtered.index)
    pca_df = pd.concat([pca_df, metadata_filtered], axis=1)
    
    # Calculate explained variance percentages
    explained_var = pca.explained_variance_ratio_ * 100
    
    # Create labels for axes
    pc1_label = f"PC1 ({explained_var[0]:.2f}%)"
    pc2_label = f"PC2 ({explained_var[1]:.2f}%)"
    
    # --- PLOTLY FIGURE ---
    fig_plotly = px.scatter(pca_df, x='PC1', y='PC2', 
                           color=classifier_col,
                           title=title_prefix,
                           labels={'PC1': pc1_label, 'PC2': pc2_label},
                           opacity=0.9)
    
    # Update plotly layout
    fig_plotly.update_traces(marker=dict(size=12))
    fig_plotly.update_layout(
        title=dict(font=dict(size=20, color="black")),
        xaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
        yaxis=dict(title_font=dict(size=18), tickfont=dict(size=16)),
        legend=dict(font=dict(size=14)),
        plot_bgcolor='white',
        width=800,
        height=600
    )
    
    # Add confidence ellipses to plotly
    for classifier in pca_df[classifier_col].unique():
        if pd.isna(classifier):
            continue
            
        subset = pca_df[pca_df[classifier_col] == classifier]
        if len(subset) < 3:  # Need at least 3 points for ellipse
            continue
            
        # Calculate ellipse parameters
        x_data = subset['PC1'].values
        y_data = subset['PC2'].values
        
        # Calculate confidence ellipse (95% confidence)
        def confidence_ellipse(x, y, n_std=2.0):
            cov = np.cov(x, y)
            pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)

            theta = np.linspace(0, 2 * np.pi, 100)
            ellipse_x = ell_radius_x * np.cos(theta)
            ellipse_y = ell_radius_y * np.sin(theta)

            # Scale and rotate
            ellipse = np.column_stack([ellipse_x, ellipse_y])
            ellipse = ellipse @ np.linalg.cholesky(cov) * n_std
            ellipse[:, 0] += np.mean(x)
            ellipse[:, 1] += np.mean(y)

            return ellipse[:, 0], ellipse[:, 1]

        try:
            ellipse_x, ellipse_y = confidence_ellipse(x_data, y_data)
            fig_plotly.add_trace(go.Scatter(
                x=ellipse_x, y=ellipse_y,
                mode='lines',
                name=f'{classifier} ellipse',
                line=dict(dash='dash'),
                showlegend=False
            ))
        except:
            pass  # Skip ellipse if calculation fails
    
    # --- MATPLOTLIB/SVG FIGURE ---
    plt.style.use('default')
    fig_mpl, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique classifiers and colors
    classifiers = pca_df[classifier_col].dropna().unique()
    colors = plt.cm.Set1(np.linspace(0, 1, len(classifiers)))
    
    # Plot points
    for i, classifier in enumerate(classifiers):
        subset = pca_df[pca_df[classifier_col] == classifier]
        ax.scatter(subset['PC1'], subset['PC2'], 
                  c=[colors[i]], label=classifier, 
                  alpha=0.9, s=80, edgecolors='black', linewidth=0.5)
        
        # Add confidence ellipse
        if len(subset) >= 3:
            try:
                x_data = subset['PC1'].values
                y_data = subset['PC2'].values
                
                # Calculate ellipse parameters
                cov = np.cov(x_data, y_data)
                lambda_, v = np.linalg.eig(cov)
                lambda_ = np.sqrt(lambda_)
                
                ellipse = Ellipse(xy=(np.mean(x_data), np.mean(y_data)),
                                width=lambda_[0]*4, height=lambda_[1]*4,
                                angle=np.rad2deg(np.arccos(v[0, 0])),
                                facecolor='none', edgecolor=colors[i],
                                alpha=0.8, linewidth=2)
                ax.add_patch(ellipse)
            except:
                pass  # Skip ellipse if calculation fails
    
    # Customize matplotlib plot
    ax.set_xlabel(pc1_label, fontsize=18, fontweight='bold')
    ax.set_ylabel(pc2_label, fontsize=18, fontweight='bold')
    ax.set_title(title_prefix, fontsize=20, fontweight='bold', color='black')
    ax.tick_params(axis='both', which='major', labelsize=16, width=2)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    # Set spine properties
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Convert matplotlib figure to SVG string
    svg_buffer = io.StringIO()
    fig_mpl.savefig(svg_buffer, format='svg', bbox_inches='tight', )
                   # facecolor='white', edgecolor='none', dpi=300)
    svg_string = svg_buffer.getvalue()
    svg_buffer.close()
    plt.close(fig_mpl)
    
    # Prepare results
    pca_results = {
        'explained_variance_ratio': explained_var,
        'components': pca.components_,
        'pca_scores': pca_df,
        'n_samples': len(pca_df),
        'n_features': features_filtered.shape[1],
        'feature_names': feature_cols
    }
    
    return {
        'plotly_fig': fig_plotly,
        'mpl_fig': fig_mpl,
        'svg_string': svg_string,
        'pca_results': pca_results
    }


if __name__ == '__main__':

    # Example usage with your data format:

    # Load your CSV data
    df = pd.read_csv('./output/food_metadata.csv')

    # Run PCA analysis
    results = create_pca_visualizations(
        data=df,
        classifier_col='Classifier',
        filter_patterns=['Omni', 'Vegan'],  # Will filter rows containing these patterns
        title_prefix="PCA NIST food readout",
        metadata_cols=['filename', 'Sample', 'description', 'Classifier', 'Sub_classifier']
    )

    # Access results:
    plotly_figure = results['plotly_fig']
    svg_string = results['svg_string']
    pca_info = results['pca_results']

    # Display plotly figure
    plotly_figure.show()

    # Save SVG to file
    with open('pca_plot.svg', 'w') as f:
        f.write(svg_string)

    # Print PCA summary
    print(f"Explained variance: PC1={pca_info['explained_variance_ratio'][0]:.2f}%, PC2={pca_info['explained_variance_ratio'][1]:.2f}%")
    print(f"Number of features used: {pca_info['n_features']}")
    print(f"Feature names: {pca_info['feature_names'][:10]}...")  # Show first 10 features
