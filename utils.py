import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pgmpy.factors.discrete.CPD import TabularCPD
from pgmpy.models import BayesianNetwork


def plot_dataframe_columns(df, figsize_base=(15, 5), bins=10, show_all_xticks=False, rwidth=0.6):
    df_notna = df.dropna()
    num_columns = len(df_notna.columns)
    
    if num_columns == 0:
        print("No columns to plot.")
        return
    
    # Calculate rows and columns for subplots
    n_cols = int(np.ceil(np.sqrt(num_columns)))
    n_rows = int(np.ceil(num_columns / n_cols))
    
    # Create subplots
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(figsize_base[0], figsize_base[1] * n_rows))
    axes = np.array(axes).flatten()
    
    # Plot each column
    for i, attr in enumerate(df_notna.columns):
        ax = axes[i]
        if np.issubdtype(df_notna[attr].dtype, np.number):
            unique_values = np.sort(df_notna[attr].unique())
            if not isinstance(bins, int) or (show_all_xticks and (len(unique_values) <= bins)):
                # Ensure bins align with unique values
                bins = np.append(unique_values, unique_values[-1] + 1) - 0.5
            
            df_notna[attr].plot(kind='hist', ax=ax, title=attr, bins=bins, edgecolor='black', rwidth=rwidth, align='mid')
            ax.set_ylabel('Frequency')
            
            if show_all_xticks:
                ax.set_xticks(unique_values)
                ax.set_xticklabels(unique_values.astype(int), rotation=0, ha='center')
                ax.set_xlim([unique_values.min() - 1, unique_values.max() + 1])
        else:
            df_notna[attr].value_counts().plot(kind='bar', ax=ax, title=attr, edgecolor='black')
            ax.set_ylabel('Count')
            ax.set_xlabel('')
            if show_all_xticks:
                ax.set_xticks(range(len(df_notna[attr].value_counts())))
    
    # Hide unused subplots
    for j in range(num_columns, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()

# We reconstruct the used encodings for acronyms used
def decode_degree(degree):
    degree_mapping = {
        "B.Pharm": "Bachelor of Pharmacy",
        "BSc": "Bachelor of Science",
        "BA": "Bachelor of Arts",
        "BCA": "Bachelor of Computer Applications",
        "M.Tech": "Master of Technology",
        "PhD": "Doctor of Philosophy",
        "Class 12": "High School Completion (12th Grade)",
        "B.Ed": "Bachelor of Education",
        "LLB": "Bachelor of Laws",
        "BE": "Bachelor of Engineering",
        "M.Ed": "Master of Education",
        "MSc": "Master of Science",
        "BHM": "Bachelor of Hotel Management",
        "M.Pharm": "Master of Pharmacy",
        "MCA": "Master of Computer Applications",
        "MA": "Master of Arts",
        "B.Com": "Bachelor of Commerce",
        "MD": "Doctor of Medicine",
        "MBA": "Master of Business Administration",
        "MBBS": "Bachelor of Medicine, Bachelor of Surgery",
        "M.Com": "Master of Commerce",
        "B.Arch": "Bachelor of Architecture",
        "LLM": "Master of Laws",
        "B.Tech": "Bachelor of Technology",
        "BBA": "Bachelor of Business Administration",
        "ME": "Master of Engineering",
        "MHM": "Master of Hotel Management",
        "Others": "Other Qualifications"
    }
    
    return degree_mapping.get(degree, "Unknown Degree")

# To follow the notation of Bayesian Networks
def to_camel_case(s: str) -> str:
    return ''.join(word.capitalize() for word in s.split())

def hierarchical_layout(G, vertical_spacing=1.5, horizontal_spacing=2.0):
    """
    Generates a hierarchical layout for a graph, distributing nodes into levels.
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Hierarchical layout works best with a Directed Acyclic Graph (DAG).")

    roots = [n for n in G.nodes() if G.in_degree(n) == 0]
    if not roots:
        shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
        avg_distance = {node: sum(d.values()) / len(d) for node, d in shortest_paths.items()}
        roots = [min(avg_distance, key=avg_distance.get)]

    levels = {}
    for root in roots:
        for node, level in nx.single_source_shortest_path_length(G, root).items():
            levels[node] = min(levels.get(node, float("inf")), level)

    level_dict = {}
    for node, level in levels.items():
        level_dict.setdefault(level, []).append(node)

    pos = {}
    for level, nodes in level_dict.items():
        x_positions = np.linspace(-len(nodes) / 2, len(nodes) / 2, len(nodes))
        for x, node in zip(x_positions, nodes):
            pos[node] = (x * horizontal_spacing, -level * vertical_spacing)

    return pos

def draw_graph(G, pos, node_color="lightblue", edge_color="gray", node_size=1000, title=None):
    """
    Draws the graph with the given layout, avoiding curved edges among same-layer nodes,
    ensuring no duplicate edges are drawn, and placing arrows at the end of edges.
    Allows setting a title that is clearly visible above the graph.
    """
    plt.figure(figsize=(15, 8))
    
    if title:
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    nx.draw(G, pos, edgelist=[], with_labels=False, node_color=node_color, edge_color=edge_color, node_size=node_size)
    
    # Track edges already drawn to avoid duplicates
    drawn_edges = set()
    
    for edge in G.edges():
        start, end = edge
        if (start, end) in drawn_edges or (end, start) in drawn_edges:
            continue
        drawn_edges.add((start, end))
        
        x1, y1 = pos[start]
        x2, y2 = pos[end]
        
        # Check if nodes are at the same level (same y position)
        curvature = 0.3 if y1 == y2 else 0.0
        plt.annotate("",
                     xy=(x2, y2), xycoords='data',
                     xytext=(x1, y1), textcoords='data',
                     arrowprops=dict(arrowstyle="->",
                                     color=edge_color,
                                     lw=1.5,
                                     connectionstyle=f"arc3,rad={curvature}"))
    
    # Improved positioning of node labels
    for node, (x, y) in pos.items():
        offset_x = 0.1 if x > 0 else -0.1
        offset_y = 0.15
        plt.text(x + offset_x, y + offset_y, str(node), fontsize=10, ha='center',
                 bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
    
    plt.show()

def print_full(cpd):
    backup = TabularCPD._truncate_strtable
    TabularCPD._truncate_strtable = lambda self, x: x
    print(cpd)
    TabularCPD._truncate_strtable = backup
    
def count_bn_parameters(model: BayesianNetwork) -> int:
    """
    Compute the total number of free parameters in a discrete Bayesian Network.
    
    For each node X with r states and parents with product of cardinalities Q,
    the number of free parameters is (r - 1) * Q.
    
    Parameters:
        model: BayesianNetwork
            The Bayesian network model with CPDs defined.
    
    Returns:
        total_parameters: int
            The total number of free parameters in the network.
    """
    total_parameters = 0
    for cpd in model.get_cpds():
        r = cpd.cardinality[0]  # Cardinality of the variable
        Q = np.prod(cpd.cardinality[1:]) if len(cpd.cardinality) > 1 else 1  # Product of parent cardinalities
        free_params = (r - 1) * Q
        total_parameters += free_params
        
    return total_parameters