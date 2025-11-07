import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

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
    parser.add_argument(
        "--plot-3d", action="store_true",
        help="Plot in 3D (requires 'umap_2' column). Falls back to 2D if missing."
    )
    parser.add_argument(
        "--show-names", action="store_true",
        help="Annotate each point with the file basename (model_path)."
    )
    parser.add_argument(
        "--name-folder", action="store_true",
        help="Use parent folder name for annotations when --show-names is set."
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


def create_color_coded_plot(df, output_dir, title, figsize, plot_3d=False, show_names=False, name_folder=False):
    """Create a color-coded scatter/3D scatter plot with optional filename annotations."""
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
    
    # Create the figure/axes
    fig = plt.figure(figsize=figsize)
    is_3d_available = plot_3d and ('umap_2' in df.columns)
    if plot_3d and not is_3d_available:
        print("Warning: 'umap_2' column not found. Falling back to 2D plot.")
    
    if is_3d_available:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    handles = []
    for i, label in enumerate(unique_labels):
        label_data = df[df['folder_label'] == label]
        color = colors[i]
        if is_3d_available:
            sc = ax.scatter(
                label_data['umap_0'],
                label_data['umap_1'],
                label_data['umap_2'],
                alpha=0.7,
                color=color,
                label=label,
                s=30,
                edgecolors='black',
                linewidth=0.3
            )
            handles.append(sc)
            # Add filename annotations
            if show_names:
                for _, row in label_data.iterrows():
                    try:
                        path_str = str(row['model_path'])
                        name = os.path.basename(os.path.dirname(path_str)) if name_folder else os.path.basename(path_str)
                    except Exception:
                        name = str(row.get('model_path', 'unknown'))
                    ax.text(row['umap_0'], row['umap_1'], row['umap_2'], name, fontsize=6)
        else:
            sc = ax.scatter(
                label_data['umap_0'],
                label_data['umap_1'],
                alpha=0.7,
                color=color,
                label=label,
                s=50,
                edgecolors='black',
                linewidth=0.5
            )
            handles.append(sc)
            if show_names:
                for _, row in label_data.iterrows():
                    try:
                        path_str = str(row['model_path'])
                        name = os.path.basename(os.path.dirname(path_str)) if name_folder else os.path.basename(path_str)
                    except Exception:
                        name = str(row.get('model_path', 'unknown'))
                    ax.annotate(name, (row['umap_0'], row['umap_1']), fontsize=6, alpha=0.8)

    # Customize the plot
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    if is_3d_available:
        ax.set_zlabel('UMAP Dimension 3', fontsize=12)
    
    # Position legend outside plot area (works for 2D; for 3D it's okay too)
    ax.legend(
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
    output_plot_path = os.path.join(output_dir, "umap_colored_3d.png" if is_3d_available else "umap_colored_2d.png")
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

    # Additionally, write an interactive HTML (3D if available, otherwise 2D)
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        color_map = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, label in enumerate(unique_labels)}
        data = []
        if is_3d_available:
            for label in unique_labels:
                sub = df[df['folder_label'] == label]
                text = [
                    (os.path.basename(os.path.dirname(str(p))) if name_folder else os.path.basename(str(p)))
                    for p in sub['model_path']
                ] if show_names else None
                data.append(go.Scatter3d(
                    x=sub['umap_0'], y=sub['umap_1'], z=sub['umap_2'],
                    mode='markers+text' if show_names else 'markers',
                    text=text,
                    textposition='top center',
                    marker=dict(size=3, color=color_map[label], line=dict(width=0.3, color='black')),
                    name=str(label)
                ))
            layout = go.Layout(
                title=title,
                scene=dict(
                    xaxis_title='UMAP Dimension 1',
                    yaxis_title='UMAP Dimension 2',
                    zaxis_title='UMAP Dimension 3',
                ),
                legend=dict(x=1.02, y=1)
            )
            fig = go.Figure(data=data, layout=layout)
            html_path = os.path.join(output_dir, 'umap_colored_3d.html')
        else:
            for label in unique_labels:
                sub = df[df['folder_label'] == label]
                text = [
                    (os.path.basename(os.path.dirname(str(p))) if name_folder else os.path.basename(str(p)))
                    for p in sub['model_path']
                ] if show_names else None
                data.append(go.Scatter(
                    x=sub['umap_0'], y=sub['umap_1'],
                    mode='markers+text' if show_names else 'markers',
                    text=text,
                    textposition='top center',
                    marker=dict(size=6, color=color_map[label], line=dict(width=0.5, color='black')),
                    name=str(label)
                ))
            layout = go.Layout(
                title=title,
                xaxis_title='UMAP Dimension 1',
                yaxis_title='UMAP Dimension 2',
                legend=dict(x=1.02, y=1)
            )
            fig = go.Figure(data=data, layout=layout)
            html_path = os.path.join(output_dir, 'umap_colored_2d.html')
        fig.write_html(html_path, include_plotlyjs='cdn')
        print(f"Saved interactive plot to {html_path}")
    except Exception as e:
        print(f"Warning: Could not save interactive HTML plot (plotly not installed or error: {e}).")


def main():
    args = parse_args()
    
    # Load the embedding data
    df = load_embedding_data(args.input_csv)
    if df is None:
        print("Failed to load embedding data. Exiting.")
        return
    
    # Create the color-coded plot
    create_color_coded_plot(
        df,
        args.output_dir,
        args.title,
        args.figsize,
        plot_3d=args.plot_3d,
        show_names=args.show_names,
        name_folder=args.name_folder,
    )
    
    print("Plot generation completed successfully!")


if __name__ == "__main__":
    main() 