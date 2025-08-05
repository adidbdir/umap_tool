import os
# Attempt to set PyOpenGL platform for headless rendering BEFORE other imports that might use OpenGL
# Try 'egl' first in NVIDIA containers, then 'osmesa' as a fallback.
# os.environ['PYOPENGL_PLATFORM'] = 'egl' 
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa' # Fallback if EGL not working or not preferred

import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State # Added State
import plotly.express as px
import pandas as pd
import sys 
import argparse
# import trimesh # No longer directly used for rendering in this version
# import io 
# import base64 
# import numpy as np 
import traceback
import subprocess # To call Blender
import time # For cache busting
import shutil # To copy a default placeholder if rendering fails or for initial setup
from PIL import Image, ImageDraw # For creating placeholder image

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Interactive UMAP Viewer with Blender Rendering")
    parser.add_argument(
        "-i", "--input-csv", 
        default=None, 
        help="Path to CSV file containing UMAP embeddings (default: outputs/test_3d/ycb_strawberry.csv)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0", 
        help="Host to run the app on (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        default=8050, 
        type=int,
        help="Port to run the app on (default: 8050)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run in debug mode"
    )
    return parser.parse_args()

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))  # Go back to project root
ASSETS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "assets"))

# Parse command line arguments
args = parse_args()
if args.input_csv:
    DEFAULT_CSV_PATH = os.path.abspath(args.input_csv)
else:
    DEFAULT_CSV_PATH = os.path.join(PROJECT_ROOT, "outputs", "test_3d", "ycb_strawberry.csv")

PLACEHOLDER_IMG_FILENAME = "placeholder.png"
RENDERED_IMAGE_PATH = os.path.join(ASSETS_DIR, PLACEHOLDER_IMG_FILENAME)
BLENDER_RENDER_SCRIPT_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "render_obj.py"))
DEFAULT_FALLBACK_PLACEHOLDER = os.path.join(ASSETS_DIR, "default_placeholder.png")

# For unique image naming to force browser refresh
RENDER_COUNTER = 0

def get_unique_render_filename():
    """Generate a unique filename for rendered images to force browser refresh."""
    global RENDER_COUNTER
    RENDER_COUNTER += 1
    return f"rendered_{RENDER_COUNTER}_{int(time.time() * 1000)}.png"

def cleanup_old_renders():
    """Clean up old render files to prevent disk space issues."""
    try:
        for file in os.listdir(ASSETS_DIR):
            if file.startswith("rendered_") and file.endswith(".png"):
                file_path = os.path.join(ASSETS_DIR, file)
                # Keep files newer than 1 hour
                if os.path.getmtime(file_path) < time.time() - 3600:
                    os.remove(file_path)
                    print(f"DEBUG: Cleaned up old render: {file}")
    except Exception as e:
        print(f"DEBUG: Error during cleanup: {e}")

# --- Ensure assets directory and a default placeholder exist ---
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    print(f"Created assets directory: {ASSETS_DIR}")

# Create a default placeholder image if it doesn't exist
if not os.path.exists(DEFAULT_FALLBACK_PLACEHOLDER):
    try:
        # Create a simple placeholder image
        img = Image.new('RGB', (400, 400), color='lightgray')
        draw = ImageDraw.Draw(img)
        draw.text((150, 190), "No Model Rendered", fill='black')
        img.save(DEFAULT_FALLBACK_PLACEHOLDER)
        print(f"Created default placeholder image at: {DEFAULT_FALLBACK_PLACEHOLDER}")
    except Exception as e:
        print(f"Error creating placeholder image: {e}")

# Copy default placeholder to rendered image path if it doesn't exist
if not os.path.exists(RENDERED_IMAGE_PATH) and os.path.exists(DEFAULT_FALLBACK_PLACEHOLDER):
    shutil.copy(DEFAULT_FALLBACK_PLACEHOLDER, RENDERED_IMAGE_PATH)
    print(f"Copied default placeholder to: {RENDERED_IMAGE_PATH}")


# --- Load Data ---
def load_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        required_columns = ['umap_0', 'umap_1', 'model_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: CSV must contain {required_columns} columns. Missing: {missing_columns}")
            return pd.DataFrame()
        
        # Check if folder_label column exists for color coding
        if 'folder_label' not in df.columns:
            print("Warning: No 'folder_label' column found. All points will have the same color.")
            df['folder_label'] = 'default'
        
        print(f"Loaded {len(df)} data points with {len(df['folder_label'].unique())} unique labels")
        print(f"Label distribution: {df['folder_label'].value_counts().to_dict()}")
        
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# --- Dash App Initialization ---
# Dash serves files from 'assets' folder automatically if it exists in the same dir as the app script
app = dash.Dash(__name__, assets_folder=ASSETS_DIR)

# --- App Layout ---
app.layout = html.Div([
    html.H1("Interactive UMAP Viewer - Blender Dynamic Rendering"),
    html.Div([
        dcc.Graph(id='umap-scatter-plot', style={'width': '60%', 'display': 'inline-block'}),
        html.Div(id='model-render-display', style={'width': '35%', 'display': 'inline-block', 'verticalAlign': 'top', 'paddingLeft': '20px'}, children=[
            # Initial src should be the placeholder in assets. Add cache-busting for updates.
            html.Img(id='rendered-model-image', 
                     src=f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={time.time()}",
                     style={'maxWidth': '100%', 'maxHeight': '400px'}),
            html.Div(id='image-status-message', style={'marginTop': '10px'}) 
        ])
    ]),
    html.Div(id='model-path-display'),
    # Store to manage Blender process state (basic attempt to prevent spamming)
    dcc.Store(id='blender-process-tracker', data={'is_rendering': False, 'last_rendered_path': None})
])

# --- Callbacks ---
@app.callback(
    Output('umap-scatter-plot', 'figure'),
    Input('umap-scatter-plot', 'id') 
)
def update_scatter_plot(_):
    df = load_data(DEFAULT_CSV_PATH)
    if df.empty:
        return {'data': [], 'layout': {'title': "No data loaded or error in data."}}
    
    # Create scatter plot with color coding by folder label
    fig = px.scatter(
        df,
        x='umap_0',
        y='umap_1',
        color='folder_label',  # Color by folder label
        custom_data=['model_path', 'folder_label'],  # Include both for hover data
        title="UMAP Embeddings (Hover to Render with Blender)",
        labels={'folder_label': 'Folder'},
        hover_data={'folder_label': True}  # Show folder label in hover
    )
    
    # Customize the layout
    fig.update_layout(
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        ),
        margin=dict(r=150)  # Make room for legend
    )
    
    return fig

@app.callback(
    [Output('rendered-model-image', 'src'),
     Output('model-path-display', 'children'),
     Output('image-status-message', 'children'),
     Output('blender-process-tracker', 'data')],
    [Input('umap-scatter-plot', 'hoverData')],
    [State('blender-process-tracker', 'data')] # Get current state of the store
)
def display_hover_data(hoverData, process_tracker_data):
    # Clean up old renders periodically
    cleanup_old_renders()
    
    # Generate unique timestamp for cache busting
    timestamp = int(time.time() * 1000)  # Use milliseconds for better uniqueness
    
    # Default image src (with cache bust)
    current_image_src = f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={timestamp}"
    model_path_text = "Hover over a point to see its 3D model render."
    status_message = "" # Initially no status message

    if not hoverData or not hoverData['points']:
        return current_image_src, model_path_text, status_message, process_tracker_data

    point_data = hoverData['points'][0]
    if 'customdata' not in point_data or not point_data['customdata']:
        model_path_text = "Error: model_path not found in hover data."
        return current_image_src, model_path_text, status_message, process_tracker_data
    
    # Extract model path and folder label from custom data
    custom_data = point_data['customdata']
    if len(custom_data) >= 2:
        obj_model_path_to_render = custom_data[0]
        folder_label = custom_data[1]
    else:
        obj_model_path_to_render = custom_data[0]
        folder_label = "unknown"
    
    model_name = os.path.basename(obj_model_path_to_render)
    model_path_text = f"Selected model: {model_name} (from {folder_label})"

    # --- Debugging output ---
    print(f"DEBUG: Hover detected for {model_name}")
    print(f"DEBUG: Model path: {obj_model_path_to_render}")
    print(f"DEBUG: Current timestamp: {timestamp}")

    # --- Blender Rendering Logic ---
    # Simplistic check: if already rendering, don't re-trigger
    if process_tracker_data.get('is_rendering'):
        status_message = f"Blender is currently rendering. Please wait."
        print("DEBUG: Rendering already in progress")
        return current_image_src, model_path_text, status_message, process_tracker_data
    
    # Check if it's the same model as last time and use cached version if available
    last_rendered_path = process_tracker_data.get('last_rendered_path')
    last_rendered_filename = process_tracker_data.get('last_rendered_filename')
    
    if (last_rendered_path == obj_model_path_to_render and 
        last_rendered_filename and 
        os.path.exists(os.path.join(ASSETS_DIR, last_rendered_filename))):
        status_message = f"Showing cached: {model_name}"
        current_image_src = f"{app.get_asset_url(last_rendered_filename)}"
        print(f"DEBUG: Using cached render for {model_name}: {last_rendered_filename}")
        return current_image_src, model_path_text, status_message, process_tracker_data

    if not os.path.exists(obj_model_path_to_render):
        status_message = f"Error: OBJ file not found at {obj_model_path_to_render}"
        print(f"DEBUG: OBJ file not found: {obj_model_path_to_render}")
        return current_image_src, model_path_text, status_message, process_tracker_data

    if not os.path.exists(BLENDER_RENDER_SCRIPT_PATH):
        status_message = "Error: Blender render script (render_obj.py) not found."
        print(f"DEBUG: Blender script not found: {BLENDER_RENDER_SCRIPT_PATH}")
        return current_image_src, model_path_text, status_message, process_tracker_data

    # --- Blender Rendering ---
    # Generate unique filename for this render
    unique_render_filename = get_unique_render_filename()
    unique_render_path = os.path.join(ASSETS_DIR, unique_render_filename)
    
    blender_executable = "blender" # Assuming 'blender' is in PATH (due to Docker setup)
    blender_command = [
        blender_executable,
        "--background",
        "--python", BLENDER_RENDER_SCRIPT_PATH,
        "--", # Argument separator for Blender's Python script
        obj_model_path_to_render,
        unique_render_path # Output path for the unique image
    ]

    process_tracker_data['is_rendering'] = True # Set flag before starting
    
    try:
        print(f"DEBUG: Starting Blender rendering...")
        print(f"DEBUG: Command: {' '.join(blender_command)}")
        print(f"DEBUG: Output path: {unique_render_path}")
        
        status_message = f"Rendering {model_name} with Blender..."
        
        # Clear previous render info before attempting new one
        process_tracker_data['last_rendered_path'] = None
        process_tracker_data['last_rendered_filename'] = None

        process = subprocess.run(blender_command, capture_output=True, text=True, timeout=30)

        print(f"DEBUG: Blender return code: {process.returncode}")

        if process.returncode == 0:
            # Check if the file was created
            if os.path.exists(unique_render_path):
                print(f"DEBUG: Blender successfully rendered: {obj_model_path_to_render}")
                status_message = f"Successfully rendered: {model_name}"
                # Use the unique filename for the image source
                current_image_src = f"{app.get_asset_url(unique_render_filename)}"
                process_tracker_data['last_rendered_path'] = obj_model_path_to_render
                process_tracker_data['last_rendered_filename'] = unique_render_filename
                print(f"DEBUG: New image src: {current_image_src}")
            else:
                print(f"DEBUG: Blender completed but file was not created: {unique_render_path}")
                status_message = f"Render completed but image not created: {model_name}"
                # Use fallback placeholder
                current_image_src = f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={timestamp}"
        else:
            print(f"DEBUG: Blender rendering failed for {obj_model_path_to_render}.")
            print(f"DEBUG: Blender stdout:\n{process.stdout}")
            print(f"DEBUG: Blender stderr:\n{process.stderr}")
            status_message = f"Failed to render {model_name}. Check console for Blender logs."
            # Use fallback placeholder
            current_image_src = f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={timestamp}"
            
    except subprocess.TimeoutExpired:
        print(f"DEBUG: Blender rendering timed out for {obj_model_path_to_render}.")
        status_message = f"Rendering {model_name} timed out."
        current_image_src = f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={timestamp}"
    except Exception as e:
        print(f"DEBUG: Exception occurred while running Blender: {e}")
        print(traceback.format_exc())
        status_message = f"Error processing {model_name}: {e}"
        current_image_src = f"{app.get_asset_url(PLACEHOLDER_IMG_FILENAME)}?t={timestamp}"
    finally:
        process_tracker_data['is_rendering'] = False # Reset flag
        print(f"DEBUG: Rendering finished, is_rendering set to False")

    print(f"DEBUG: Final image src: {current_image_src}")
    return current_image_src, model_path_text, status_message, process_tracker_data

# --- Run Application ---
if __name__ == '__main__':
    print(f"Using CSV file: {DEFAULT_CSV_PATH}")
    
    if not os.path.exists(DEFAULT_CSV_PATH):
        print(f"Error: The data file {DEFAULT_CSV_PATH} was not found.")
        print("Please run test_3d.py first to generate the embeddings and model paths.")
        print("Or specify a different CSV file with: python interactive_umap_viewer.py -i /path/to/your/file.csv")
        sys.exit(1)
    
    if not os.path.exists(BLENDER_RENDER_SCRIPT_PATH):
        print(f"Error: The Blender rendering script {BLENDER_RENDER_SCRIPT_PATH} was not found.")
        sys.exit(1)

    # Check if blender executable is available (basic check)
    try:
        subprocess.run(["blender", "--version"], capture_output=True, check=True)
        print("Blender executable found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: 'blender' command not found in PATH. Please ensure Blender is installed and in PATH.")
        sys.exit(1)

    print(f"Starting Dash app on {args.host}:{args.port}")
    print(f"Images will be rendered to: {RENDERED_IMAGE_PATH}")
    print(f"Hover over points to trigger Blender rendering.")
    print(f"Open your browser and go to: http://{args.host}:{args.port}")
    
    app.run(debug=args.debug, host=args.host, port=args.port) 