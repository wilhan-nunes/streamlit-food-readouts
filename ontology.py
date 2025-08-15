import yaml
import pandas as pd
from collections import defaultdict, Counter
import plotly.graph_objects as go


def load_ontology(yaml_file):
    """Load the ontology from YAML file"""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def extract_hierarchy_paths(ontology, current_path=None, target_level=None):
    """Extract all possible paths from root to leaf in the ontology

    Args:
        ontology: The ontology dictionary
        current_path: Current path being built (for recursion)
        target_level: Desired level to extract (1-5). If None, extracts all paths to leaves
    """
    if current_path is None:
        current_path = []

    paths = []

    if isinstance(ontology, dict):
        for key, value in ontology.items():
            new_path = current_path + [key]

            # If we've reached the target level, add this path
            if target_level is not None and len(new_path) == target_level:
                paths.append(new_path)
            elif isinstance(value, dict):
                # Continue recursing if we haven't reached target level or if no target level specified
                if target_level is None or len(new_path) < target_level:
                    paths.extend(extract_hierarchy_paths(value, new_path, target_level))
            else:
                # This is a leaf node - add it if no target level specified
                if target_level is None:
                    paths.append(new_path)

    return paths


def build_classification_map(ontology, target_level=5):
    """Build a mapping from each item to its full hierarchical path

    Args:
        ontology: The ontology dictionary
        target_level: The level to extract items from (1-5)
    """
    paths = extract_hierarchy_paths(ontology, target_level=target_level)
    classification_map = {}

    for path in paths:
        if len(path) >= target_level:  # Ensure we have enough levels
            item = path[-1]  # The item at the target level

            # Build classification for all levels up to the target
            classification = {}
            for i in range(min(len(path), 5)):  # Cap at 5 levels
                classification[f'level_{i + 1}'] = path[i]

            classification_map[item] = classification

    return classification_map


def find_matching_columns(df, classification_map):
    """Find TSV columns that match ontology items"""
    ontology_items = set(classification_map.keys())
    df_columns = set(df.columns)

    # Direct matches
    direct_matches = ontology_items.intersection(df_columns)

    return direct_matches


def count_classifications_by_level(df, classification_map, matched_columns, peak_threshold=0):
    """Count classifications at each hierarchical level"""
    # Determine max level available in classification_map
    max_level = 1
    if classification_map:
        sample_item = next(iter(classification_map.values()))
        max_level = max([int(k.split('_')[1]) for k in sample_item.keys() if k.startswith('level_')])

    # Initialize level_counts only for available levels
    level_counts = {}
    for i in range(1, max_level + 1):
        level_counts[f'level_{i}'] = Counter()

    # Process each row in the dataframe
    for idx, row in df.iterrows():
        # Initialize row_classifications only for available levels
        row_classifications = {}
        for i in range(1, max_level + 1):
            row_classifications[f'level_{i}'] = set()

        # Check each matched column
        for col in matched_columns:
            if col in df.columns:
                value = row[col]

                # Only count if value is significant - defined by threshold
                if pd.notna(value) and value > peak_threshold:
                    # Get the ontology item this column maps to
                    ontology_item = col #matched_columns[col] if col in matched_columns else col

                    if ontology_item in classification_map:
                        hierarchy = classification_map[ontology_item]

                        # Add to each available level (avoid double counting in the same row)
                        for level_key in hierarchy.keys():
                            if level_key in row_classifications:
                                row_classifications[level_key].add(hierarchy[level_key])

        # Count unique classifications per row at each level
        for level in row_classifications:
            for classification in row_classifications[level]:
                level_counts[level][classification] += 1

    return level_counts


def create_sankey_plot(df, classification_map, matched_columns, target_level=5, min_count=1, peak_threshold=0):
    """Create a Sankey diagram showing the flow through hierarchical levels

    Args:
        df: DataFrame with the data
        classification_map: Mapping of items to their hierarchical classifications
        matched_columns: Dictionary of matched columns
        target_level: Maximum level to include in the plot
        min_count: Minimum count to include a node/flow
    """

    # Collect all flows between levels
    flows = defaultdict(int)
    all_nodes = set()

    # Process each row to build flows
    for idx, row in df.iterrows():
        # Track what classifications this row contributes to
        row_paths = []

        for col in matched_columns:
            if col in df.columns:
                value = row[col]

                if pd.notna(value) and value > peak_threshold:
                    ontology_item = col

                    if ontology_item in classification_map:
                        hierarchy = classification_map[ontology_item]

                        # Build the path for this item
                        path = []
                        for i in range(1, target_level + 1):
                            level_key = f'level_{i}'
                            if level_key in hierarchy:
                                path.append(f"L{i}: {hierarchy[level_key]}".title())

                        if len(path) > 1:
                            row_paths.append(path)

        # Add flows for each path in this row
        for path in row_paths:
            # Add all nodes
            all_nodes.update(path)

            # Add flows between consecutive levels
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                flows[(source, target)] += 1

    # Filter flows by minimum count
    flows = {k: v for k, v in flows.items() if v >= min_count}

    if not flows:
        print("No flows found. Try reducing min_count parameter.")
        return None

    # Create node list
    nodes = list(all_nodes)
    node_dict = {node: i for i, node in enumerate(nodes)}

    # Prepare data for Sankey
    source = []
    target = []
    value = []

    for (src, tgt), count in flows.items():
        if src in node_dict and tgt in node_dict:
            source.append(node_dict[src])
            target.append(node_dict[tgt])
            value.append(count)

    # Create colors for different levels
    colors = [
        'rgba(31, 119, 180, 0.8)',  # Level 1 - blue
        'rgba(255, 127, 14, 0.8)',  # Level 2 - orange
        'rgba(44, 160, 44, 0.8)',  # Level 3 - green
        'rgba(214, 39, 40, 0.8)',  # Level 4 - red
        'rgba(148, 103, 189, 0.8)',  # Level 5 - purple
    ]

    # Assign colors based on level
    node_colors = []
    for node in nodes:
        level = int(node.split(':')[0][1:])  # Extract level number
        color_idx = (level - 1) % len(colors)
        node_colors.append(colors[color_idx])

    # Create the Sankey diagram
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes,
            color=node_colors,
        ),
        link=dict(
            source=source,
            target=target,
            value=value,
            color='rgba(100, 100, 100, 0.3)',
        ),
        textfont=dict(
            size=18,
            color="black",
            shadow="0px -0px 2px white"),
        arrangement='snap',
    )])

    fig.update_layout(
        title_text=f"Ontology Classification Flow (Levels 1-{target_level})",
        font_size=10,
        width=1200,
        height=800
    )

    return fig

def add_sankey_ontology(ontology, df, target_level=4, peak_threshold=0):
    """Main function to add Sankey plot based on ontology and dataframe"""
    classification_map = build_classification_map(ontology, target_level=target_level)
    matches = find_matching_columns(df, classification_map)
    level_counts = count_classifications_by_level(df, classification_map, matches, peak_threshold=peak_threshold)
    fig = create_sankey_plot(df, classification_map, matches, target_level=target_level, peak_threshold=peak_threshold)
    return fig


if __name__ == '__main__':

    df = pd.read_csv('/Users/wilhan/Downloads/food_metadata (2).tsv', sep='\t')
    ontology = load_ontology(
        'food_ontology.yaml')

    target_level = 3
    peak_threshold = 0

    paths = extract_hierarchy_paths(ontology)
    classification_map = build_classification_map(ontology, target_level=target_level)
    matches = find_matching_columns(df, classification_map)
    level_counts = count_classifications_by_level(df, classification_map, matches, peak_threshold=peak_threshold)
    fig = create_sankey_plot(df, classification_map, matches, target_level=target_level, peak_threshold=peak_threshold)
    fig.show()