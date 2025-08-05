import bpy
import sys
import os
from mathutils import Vector


def clear_scene():
    """Remove all existing objects, cameras, and lights from the current scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Purge orphan data blocks to free memory (Blender 3.0+). This operator requires an Outliner
    # context, which is unavailable in background mode. Therefore, attempt the call but ignore
    # failures to maintain compatibility across Blender versions and execution modes.
    if hasattr(bpy.ops.outliner, "orphan_purge"):
        try:
            bpy.ops.outliner.orphan_purge(do_recursive=True)
        except Exception:
            # Safe to skip if context is missing.
            pass


def import_obj(obj_path: str):
    """Import an OBJ file. Supports both new and legacy import operators."""
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"OBJ file not found: {obj_path}")

    try:
        bpy.ops.wm.obj_import(filepath=obj_path)
    except AttributeError:
        # Fallback for older Blender versions
        bpy.ops.import_scene.obj(filepath=obj_path)



def compute_bounds(objects):
    """Return the center point and the maximum dimension length of all mesh objects."""
    bbox_corners = []
    for obj in objects:
        for corner in obj.bound_box:
            bbox_corners.append(obj.matrix_world @ Vector(corner))

    if not bbox_corners:
        return Vector((0, 0, 0)), 1.0

    min_coord = Vector((min(c.x for c in bbox_corners),
                        min(c.y for c in bbox_corners),
                        min(c.z for c in bbox_corners)))
    max_coord = Vector((max(c.x for c in bbox_corners),
                        max(c.y for c in bbox_corners),
                        max(c.z for c in bbox_corners)))

    center = (min_coord + max_coord) / 2
    max_dim = max((max_coord - min_coord))
    return center, max_dim


def setup_lights(center: Vector, size: float):
    """Create a simple three-point lighting setup."""
    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == "LIGHT":
            bpy.data.objects.remove(obj, do_unlink=True)

    # Key, Fill, Back lights (SUN type for even illumination)
    bpy.ops.object.light_add(type="SUN", location=center + Vector((size * 2, -size * 2, size * 2)))
    bpy.context.object.data.energy = 3.0

    bpy.ops.object.light_add(type="SUN", location=center + Vector((-size * 2, -size * 1.5, size * 1.5)))
    bpy.context.object.data.energy = 1.5

    bpy.ops.object.light_add(type="SUN", location=center + Vector((0, size * 2, size)))
    bpy.context.object.data.energy = 2.0


def ensure_materials(objects):
    """Give mesh objects a basic Principled BSDF material if they have none."""
    for obj in objects:
        if obj.type == "MESH" and not obj.data.materials:
            mat = bpy.data.materials.new(name=f"Material_{obj.name}")
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.inputs["Base Color"].default_value = (0.8, 0.8, 0.8, 1.0)
            mat.node_tree.links.new(bsdf.outputs["BSDF"],
                                     mat.node_tree.nodes.new(type="ShaderNodeOutputMaterial").inputs["Surface"])
            obj.data.materials.append(mat)


def setup_camera(center: Vector, distance: float):
    """Create and return a camera object positioned at the origin; we will move it per-view."""
    # Remove existing cameras
    for cam in [obj for obj in bpy.data.objects if obj.type == "CAMERA"]:
        bpy.data.objects.remove(cam, do_unlink=True)

    bpy.ops.object.camera_add(location=(0, 0, 0))
    cam = bpy.context.object
    cam.name = "Camera"
    bpy.context.scene.camera = cam
    return cam


def render_view(cam, center: Vector, location: Vector, output_filepath: str):
    """Render a single view from the specified camera location."""
    cam.location = location
    direction = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    scene = bpy.context.scene
    scene.render.image_settings.file_format = 'PNG'
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  # adjust for quality/performance
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.filepath = output_filepath

    bpy.ops.render.render(write_still=True)

    if not os.path.exists(output_filepath):
        raise RuntimeError(f"Render failed, file not found: {output_filepath}")


def render_four_views(obj_path: str, output_dir: str):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    clear_scene()
    import_obj(obj_path)

    # Collect mesh objects
    mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    center, max_dim = compute_bounds(mesh_objects)

    ensure_materials(mesh_objects)
    setup_lights(center, max_dim)

    cam = setup_camera(center, max_dim)

    # Distance of camera from object center
    cam_dist = max_dim * 3.5  # was 2.5; increased for better framing

    z_offset = max_dim * 0.2  # raise camera a bit so we view slightly from above

    # Define camera positions relative to object center
    views = {
        "1_front": Vector((0, -cam_dist, z_offset)),
        "2_back": Vector((0, cam_dist, z_offset)),
        "3_left": Vector((cam_dist, 0, z_offset)),
        "4_right": Vector((-cam_dist, 0, z_offset)),
    }

    for name, loc in views.items():
        output_path = os.path.join(output_dir, f"{name}.png")
        print(f"Rendering view '{name}' to {output_path}")
        render_view(cam, center, center + loc, output_path)

    print("All four views rendered successfully.")


if __name__ == "__main__":
    args = sys.argv
    if "--" not in args:
        print("Usage: blender --background --python render_obj_four_views.py -- <path_to_obj> <output_dir>")
        sys.exit(1)

    idx = args.index("--") + 1
    if len(args) - idx < 2:
        print("Usage: blender --background --python render_obj_four_views.py -- <path_to_obj> <output_dir>")
        sys.exit(1)

    obj_file = args[idx]
    out_dir = args[idx + 1]

    try:
        render_four_views(obj_file, out_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1) 