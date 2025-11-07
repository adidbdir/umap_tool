import bpy
import sys
import os
import csv
import math
from mathutils import Vector

"""
Usage (example):
blender --background --python render_obj_multi_views.py -- model.obj output_dir 12
→ 12 等間隔のビューを出力し、角度とファイル名の対応を angles.csv に保存します。
"""


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    if hasattr(bpy.ops.outliner, "orphan_purge"):
        try:
            bpy.ops.outliner.orphan_purge(do_recursive=True)
        except Exception:
            pass


def import_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ext = os.path.splitext(path)[1].lower()
    # OBJ
    if ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=path)
        return
    # GLB/GLTF
    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
        return
    # PLY
    if ext == ".ply":
        try:
            bpy.ops.wm.ply_import(filepath=path)
        except Exception:
            bpy.ops.import_mesh.ply(filepath=path)
        return
    # STL
    if ext == ".stl":
        try:
            bpy.ops.wm.stl_import(filepath=path)
        except Exception:
            bpy.ops.import_mesh.stl(filepath=path)
        return
    # FBX
    if ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
        return
    # DAE (Collada)
    if ext == ".dae":
        try:
            bpy.ops.wm.collada_import(filepath=path)
        except Exception:
            # Some Blender versions may expose it under import_scene
            bpy.ops.import_scene.dae(filepath=path)
        return
    raise ValueError(f"Unsupported extension for import: {ext}")


def compute_bounds():
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        return Vector((0, 0, 0)), 1.0
    corners = [o.matrix_world @ Vector(c) for o in mesh_objs for c in o.bound_box]
    min_c = Vector((min(c.x for c in corners), min(c.y for c in corners), min(c.z for c in corners)))
    max_c = Vector((max(c.x for c in corners), max(c.y for c in corners), max(c.z for c in corners)))
    center = (min_c + max_c) / 2
    max_dim = max(max_c - min_c)
    return center, max_dim


def ensure_materials():
    for o in bpy.context.scene.objects:
        if o.type == "MESH" and not o.data.materials:
            m = bpy.data.materials.new(name=f"Mat_{o.name}")
            m.use_nodes = True
            bsdf = m.node_tree.nodes.new("ShaderNodeBsdfPrincipled")
            m.node_tree.links.new(bsdf.outputs["BSDF"], m.node_tree.nodes.new("ShaderNodeOutputMaterial").inputs["Surface"])
            o.data.materials.append(m)


def setup_lights(center: Vector, size: float):
    for obj in [o for o in bpy.data.objects if o.type == "LIGHT"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.object.light_add(type="SUN", location=center + Vector((size*2, -size*2, size*2)))
    bpy.context.object.data.energy = 3.0
    bpy.ops.object.light_add(type="SUN", location=center + Vector((-size*2, -size*1.5, size*1.5)))
    bpy.context.object.data.energy = 1.5
    bpy.ops.object.light_add(type="SUN", location=center + Vector((0, size*2, size)))
    bpy.context.object.data.energy = 2.0


def setup_camera():
    for c in [o for o in bpy.data.objects if o.type == "CAMERA"]:
        bpy.data.objects.remove(c, do_unlink=True)
    bpy.ops.object.camera_add(location=(0,0,0))
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    return cam


def render_single(cam, center: Vector, loc: Vector, filepath: str):
    cam.location = loc
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = 64
    scn.render.resolution_x = 800
    scn.render.resolution_y = 600
    scn.render.image_settings.file_format = 'PNG'
    scn.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    if not os.path.exists(filepath):
        raise RuntimeError(f"Render failed: {filepath}")


def render_views(obj_path: str, out_dir: str, num_views: int):
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    import_model(obj_path)

    center, max_dim = compute_bounds()
    ensure_materials()
    setup_lights(center, max_dim)
    cam = setup_camera()

    cam_dist = max_dim * 3.5
    z_offset = max_dim * 0.2

    csv_rows = [("filename", "angle_deg")]

    for idx in range(num_views):
        angle = 360.0 * idx / num_views  # degrees
        rad = math.radians(angle)
        x = cam_dist * math.sin(rad)
        y = -cam_dist * math.cos(rad)  # front (-Y) is 0 deg
        loc = center + Vector((x, y, z_offset))
        fname = f"view_{idx:03d}_{int(angle)}deg.png"
        fpath = os.path.join(out_dir, fname)
        print(f"Rendering {fname} @ {angle:.1f} deg")
        render_single(cam, center, loc, fpath)
        csv_rows.append((fname, f"{angle:.2f}"))

    csv_path = os.path.join(out_dir, "angles.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"Completed {num_views} views. CSV saved to {csv_path}")


if __name__ == "__main__":
    if "--" not in sys.argv:
        print("Usage: blender --background --python render_obj_multi_views.py -- <obj_path> <output_dir> <num_views>")
        sys.exit(1)
    idx = sys.argv.index("--") + 1
    if len(sys.argv) - idx < 3:
        print("Usage: blender --background --python render_obj_multi_views.py -- <obj_path> <output_dir> <num_views>")
        sys.exit(1)
    obj_file = sys.argv[idx]
    out_dir = sys.argv[idx+1]
    try:
        n_views = int(sys.argv[idx+2])
    except ValueError:
        print("<num_views> must be integer")
        sys.exit(1)
    render_views(obj_file, out_dir, n_views) 