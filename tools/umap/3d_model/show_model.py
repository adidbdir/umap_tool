import trimesh
import sys

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        try:
            mesh = trimesh.load_mesh(model_path)
            if isinstance(mesh, trimesh.Scene):
                 # If it's a scene, try to convert to a single mesh for simplicity
                mesh = mesh.dump(concatenate=True)
            if mesh and (hasattr(mesh, 'vertices') and len(mesh.vertices) > 0):
                mesh.show()
            else:
                print(f"Error: Could not load or no geometry found in {model_path}")
        except Exception as e:
            print(f"Error showing model {model_path}: {e}")
    else:
        print("Usage: python show_model.py <path_to_model_file>") 