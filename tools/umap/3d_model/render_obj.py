import bpy
import sys
import os
import math
from mathutils import Vector

def render_model(obj_file_path, output_image_path):
    # Ensure output directory exists
    output_dir = os.path.dirname(output_image_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Clear existing objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Import OBJ - Updated for Blender 4.0+
    try:
        # For Blender 4.0+, use the new wavefront_obj import
        bpy.ops.wm.obj_import(filepath=obj_file_path)
        print(f"Successfully imported OBJ: {obj_file_path}")
    except AttributeError:
        try:
            # Fallback for older Blender versions
            bpy.ops.import_scene.obj(filepath=obj_file_path)
            print(f"Successfully imported OBJ (legacy): {obj_file_path}")
        except AttributeError:
            print(f"Error: No suitable OBJ import operator found for {obj_file_path}")
            sys.exit(1)
    except RuntimeError as e:
        print(f"Error importing OBJ {obj_file_path}: {e}")
        sys.exit(1)

    # Get imported objects
    imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
    
    if not imported_objects:
        print(f"No mesh objects imported from {obj_file_path}")
        return

    # Select all imported objects
    bpy.ops.object.select_all(action='DESELECT')
    for obj in imported_objects:
        obj.select_set(True)
    
    if imported_objects:
        bpy.context.view_layer.objects.active = imported_objects[0]
        
        # Calculate bounding box for better camera positioning
        bbox_corners = []
        for obj in imported_objects:
            for corner in obj.bound_box:
                bbox_corners.append(obj.matrix_world @ Vector(corner))
        
        if bbox_corners:
            # Calculate center and dimensions
            min_coord = Vector((min(corner.x for corner in bbox_corners),
                              min(corner.y for corner in bbox_corners),
                              min(corner.z for corner in bbox_corners)))
            max_coord = Vector((max(corner.x for corner in bbox_corners),
                              max(corner.y for corner in bbox_corners),
                              max(corner.z for corner in bbox_corners)))
            center = (min_coord + max_coord) / 2
            dimensions = max_coord - min_coord
            max_dim = max(dimensions)
            print(f"Object center: {center}, max dimension: {max_dim}")
        else:
            center = Vector((0, 0, 0))
            max_dim = 1.0

    # --- Setup Camera ---
    # Remove existing cameras
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj, do_unlink=True)
    
    # Add new camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.object
    camera.name = "Camera"
    
    # Position camera for good view
    camera_distance = max_dim * 3
    camera.location = center + Vector((camera_distance * 0.7, -camera_distance * 0.7, camera_distance * 0.5))
    
    # Point camera at the center
    direction = center - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    
    bpy.context.scene.camera = camera

    # --- Setup Lighting ---
    # Remove existing lights
    for obj in list(bpy.data.objects):
        if obj.type == 'LIGHT':
            bpy.data.objects.remove(obj, do_unlink=True)

    # Add three-point lighting
    # Key light (main light)
    bpy.ops.object.light_add(type='SUN', location=center + Vector((max_dim * 2, -max_dim * 2, max_dim * 2)))
    key_light = bpy.context.object
    key_light.name = "KeyLight"
    key_light.data.energy = 3.0
    
    # Fill light (softer light)
    bpy.ops.object.light_add(type='SUN', location=center + Vector((-max_dim * 2, -max_dim * 1.5, max_dim * 1.5)))
    fill_light = bpy.context.object
    fill_light.name = "FillLight"
    fill_light.data.energy = 1.5

    # Back light (rim light)
    bpy.ops.object.light_add(type='SUN', location=center + Vector((0, max_dim * 2, max_dim)))
    back_light = bpy.context.object
    back_light.name = "BackLight"
    back_light.data.energy = 2.0

    # --- Setup Rendering ---
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    scene.cycles.samples = 64  # Reduced for faster rendering
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = output_image_path
    scene.render.resolution_x = 800
    scene.render.resolution_y = 600
    scene.render.film_transparent = True

    # Set viewport shading to material preview for better lighting
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'MATERIAL'

    # Add basic materials to objects without materials
    for obj in imported_objects:
        if obj.type == 'MESH' and not obj.data.materials:
            # Create a simple material
            mat = bpy.data.materials.new(name=f"Material_{obj.name}")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            
            # Clear default nodes
            nodes.clear()
            
            # Add Principled BSDF
            bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
            bsdf.inputs['Base Color'].default_value = (0.8, 0.8, 0.8, 1.0)
            bsdf.inputs['Metallic'].default_value = 0.0
            bsdf.inputs['Roughness'].default_value = 0.3
            
            # Add output node
            output = nodes.new(type='ShaderNodeOutputMaterial')
            
            # Link nodes
            mat.node_tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
            
            # Assign material to object
            obj.data.materials.append(mat)

    # Render
    try:
        print(f"Starting render to: {output_image_path}")
        bpy.ops.render.render(write_still=True)
        print(f"Blender rendered: {obj_file_path} to {output_image_path}")
        
        # Verify file was created
        if os.path.exists(output_image_path):
            print(f"Render successful: {output_image_path} created")
        else:
            print(f"Error: Render file not created at {output_image_path}")
            sys.exit(1)
            
    except RuntimeError as e:
        print(f"Error during Blender render: {e}")
        sys.exit(1)

if __name__ == "__main__":
    args = sys.argv
    if "--" not in args:
        print("Blender script error: No arguments provided after --")
        sys.exit(1)
        
    script_args_index = args.index("--") + 1
    
    if len(args) - script_args_index < 2:
        print("Usage: blender --background --python render_obj.py -- <path_to_obj> <output_image_path>")
        sys.exit(1)
        
    obj_path_arg = args[script_args_index]
    output_path_arg = args[script_args_index + 1]
    
    print(f"Rendering {obj_path_arg} to {output_path_arg}")
    render_model(obj_path_arg, output_path_arg) 