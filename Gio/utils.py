import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_dataframe_columns(df, figsize_base=(15, 5), bins=10, show_all_xticks=False, rwidth=0.6):
    """
    Generates histograms for numerical columns and bar plots for categorical columns in a DataFrame.
    
    Parameters:
    - df: pd.DataFrame, the DataFrame containing the data.
    - figsize_base: tuple, base size of the figure (width, height multiplier for rows).
    - bins: int, number of bins for histogram plots.
    - show_all_xticks: bool, whether to show all x-tick labels for categorical variables.
    """
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