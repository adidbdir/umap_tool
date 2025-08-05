import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input-csv", required=True, type=str,
        help="Path to CSV file containing UMAP embeddings and folder labels"
    )
    parser.add_argument(
        "-o", "--output-dir", default="outputs/plot_3d", type=str,
        help="Directory to save the output plot"
    )
    parser.add_argument(
        "--title", default="3D Model UMAP Embeddings", type=str,
        help="Title for the plot"
    )
    parser.add_argument(
        "--figsize", nargs=2, type=float, default=[12, 8],
        help="Figure size (width height)"
    )
    return parser.parse_args()


def load_embedding_data(csv_path: str):
    """Load embedding data from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        # Check required columns
        required_columns = ['umap_0', 'umap_1', 'model_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: CSV must contain {required_columns} columns. Missing: {missing_columns}")
            return None
        
        # Add default folder label if not present
        if 'folder_label' not in df.columns:
            print("Warning: No 'folder_label' column found. Using default label.")
            df['folder_label'] = 'default'
        
        print(f"Loaded {len(df)} data points with {len(df['folder_label'].unique())} unique labels")
        print(f"Label distribution: {df['folder_label'].value_counts().to_dict()}")
        
        return df
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None


def create_color_coded_plot(df, output_dir, title, figsize):
    """Create a color-coded scatter plot based on folder labels."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Get unique labels and assign colors
    unique_labels = sorted(df['folder_label'].unique())
    num_labels = len(unique_labels)
    
    # Use a colormap with enough distinct colors
    if num_labels <= 10:
        cmap = plt.get_cmap("tab10")
    elif num_labels <= 20:
        cmap = plt.get_cmap("tab20")
    else:
        cmap = plt.get_cmap("viridis")
    
    colors = [cmap(i / max(num_labels - 1, 1)) for i in range(num_labels)]
    
    # Create the plot
    plt.figure(figsize=figsize)
    
    handles = []
    for i, label in enumerate(unique_labels):
        label_data = df[df['folder_label'] == label]
        
        scatter = plt.scatter(
            label_data['umap_0'],
            label_data['umap_1'],
            alpha=0.7,
            color=colors[i],
            label=label,
            s=50,
            edgecolors='black',
            linewidth=0.5
        )
        handles.append(scatter)
    
    # Customize the plot
    plt.title(title, fontsize=16, pad=20)
    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    
    # Position legend outside plot area
    plt.legend(
        handles=handles,
        labels=unique_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        title='Folder Labels',
        title_fontsize=12
    )
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_plot_path = os.path.join(output_dir, "umap_3d_colored.png")
    plt.savefig(output_plot_path, format="png", dpi=300, bbox_inches='tight')
    print(f"Saved color-coded plot to {output_plot_path}")
    
    # Also save a summary
    summary_path = os.path.join(output_dir, "plot_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"UMAP 3D Plot Summary\n")
        f.write(f"====================\n\n")
        f.write(f"Total data points: {len(df)}\n")
        f.write(f"Number of categories: {num_labels}\n\n")
        f.write(f"Label distribution:\n")
        for label in unique_labels:
            count = len(df[df['folder_label'] == label])
            percentage = (count / len(df)) * 100
            f.write(f"  {label}: {count} points ({percentage:.1f}%)\n")
    
    print(f"Saved plot summary to {summary_path}")
    
    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass  # In case we're in a non-interactive environment


def main():
    args = parse_args()
    
    # Load the embedding data
    df = load_embedding_data(args.input_csv)
    if df is None:
        print("Failed to load embedding data. Exiting.")
        return
    
    # Create the color-coded plot
    create_color_coded_plot(df, args.output_dir, args.title, args.figsize)
    
    print("Plot generation completed successfully!")


if __name__ == "__main__":
    main() 