IMAGE_PATH = "/home/mrunal/omni_drones/OmniDrones/omni_drones/robots/assets/gate/textures/bitmap.png"
USD_OUTPUT_PATH = "/home/mrunal/Downloads/output_gate.usd"

import bpy
import bmesh
import os

# =========================================================
# USER PARAMETERS
# =========================================================
OUTER_WIDTH = 2.5          # along Y
OUTER_HEIGHT = 2.5         # along Z
THICKNESS = 0.25           # along X (out of page)
FRAME_WIDTH = 0.5          # border thickness on all 4 sides

ROOT_NAME = "gate"
GRAY_MAT_NAME = "GateGrayMaterial"
OVERLAY_MAT_NAME = "GateFrontBackOverlay"

BASE_GRAY = (0.5, 0.5, 0.5, 1.0)

# =========================================================
# VALIDATION
# =========================================================
INNER_WIDTH = OUTER_WIDTH - 2.0 * FRAME_WIDTH
INNER_HEIGHT = OUTER_HEIGHT - 2.0 * FRAME_WIDTH

if INNER_WIDTH <= 0 or INNER_HEIGHT <= 0:
    raise ValueError("FRAME_WIDTH is too large; inner opening becomes non-positive.")

if not os.path.isfile(IMAGE_PATH):
    raise FileNotFoundError(f"PNG not found: {IMAGE_PATH}")

# =========================================================
# CLEANUP
# =========================================================
def remove_object_recursive(obj):
    for child in list(obj.children):
        remove_object_recursive(child)
    if obj.type == "MESH" and obj.data is not None:
        mesh_data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh_data.users == 0:
            bpy.data.meshes.remove(mesh_data, do_unlink=True)
    else:
        bpy.data.objects.remove(obj, do_unlink=True)

if ROOT_NAME in bpy.data.objects:
    remove_object_recursive(bpy.data.objects[ROOT_NAME])

for name in ["gate_top", "gate_bottom", "gate_left", "gate_right"]:
    if name in bpy.data.objects:
        remove_object_recursive(bpy.data.objects[name])

for mat_name in [GRAY_MAT_NAME, OVERLAY_MAT_NAME]:
    if mat_name in bpy.data.materials:
        bpy.data.materials.remove(bpy.data.materials[mat_name], do_unlink=True)

bpy.ops.object.select_all(action='DESELECT')

# =========================================================
# GLOBAL COORDINATE CONVENTION
# X = thickness (out of page)
# Y = horizontal
# Z = vertical
#
# Root origin:
#   x = 0   -> back face plane
#   y = 0   -> centered horizontally
#   z = 0   -> bottom center
# =========================================================
x_back = -THICKNESS / 2.0
x_front =  THICKNESS / 2.0

y_outer_min = -OUTER_WIDTH / 2.0
y_outer_max =  OUTER_WIDTH / 2.0
y_inner_min = -INNER_WIDTH / 2.0
y_inner_max =  INNER_WIDTH / 2.0

z_outer_min = 0.0
z_outer_max = OUTER_HEIGHT
z_inner_min = FRAME_WIDTH
z_inner_max = OUTER_HEIGHT - FRAME_WIDTH

EPS = 1e-6

# =========================================================
# CREATE MATERIALS
# =========================================================
gray_mat = bpy.data.materials.new(name=GRAY_MAT_NAME)
gray_mat.use_nodes = True
nodes = gray_mat.node_tree.nodes
links = gray_mat.node_tree.links
for n in list(nodes):
    nodes.remove(n)

gray_out = nodes.new(type="ShaderNodeOutputMaterial")
gray_out.location = (250, 0)

gray_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
gray_bsdf.location = (0, 0)
gray_bsdf.inputs["Base Color"].default_value = BASE_GRAY
links.new(gray_bsdf.outputs["BSDF"], gray_out.inputs["Surface"])

overlay_mat = bpy.data.materials.new(name=OVERLAY_MAT_NAME)
overlay_mat.use_nodes = True
nodes = overlay_mat.node_tree.nodes
links = overlay_mat.node_tree.links
for n in list(nodes):
    nodes.remove(n)

overlay_out = nodes.new(type="ShaderNodeOutputMaterial")
overlay_out.location = (500, 0)

overlay_bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
overlay_bsdf.location = (250, 0)

gray_rgb = nodes.new(type="ShaderNodeRGB")
gray_rgb.location = (-400, 100)
gray_rgb.outputs["Color"].default_value = BASE_GRAY

img = nodes.new(type="ShaderNodeTexImage")
img.location = (-400, -100)
img.image = bpy.data.images.load(IMAGE_PATH)

mix = nodes.new(type="ShaderNodeMixRGB")
mix.location = (0, 0)
mix.blend_type = 'MIX'

links.new(gray_rgb.outputs["Color"], mix.inputs["Color1"])
links.new(img.outputs["Color"], mix.inputs["Color2"])
links.new(img.outputs["Alpha"], mix.inputs["Fac"])

links.new(mix.outputs["Color"], overlay_bsdf.inputs["Base Color"])
links.new(overlay_bsdf.outputs["BSDF"], overlay_out.inputs["Surface"])

# =========================================================
# CREATE ROOT EMPTY
# =========================================================
root = bpy.data.objects.new(ROOT_NAME, None)
root.empty_display_type = 'PLAIN_AXES'
root.location = (0.0, 0.0, 0.0)
bpy.context.collection.objects.link(root)

# =========================================================
# HELPERS
# =========================================================
def create_box_part(name, y0, y1, z0, z1):
    """
    Creates one rectangular box occupying:
      x in [x_back, x_front]
      y in [y0, y1]
      z in [z0, z1]
    with UVs on front/back mapped in GLOBAL gate YZ coordinates,
    so the image spans continuously across all 4 pieces.
    """
    mesh = bpy.data.meshes.new(name + "_mesh")
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = root

    bm = bmesh.new()

    def v(x, y, z):
        return bm.verts.new((x, y, z))

    # back face (x = x_back)
    b_bl = v(x_back,  y0, z0)
    b_br = v(x_back,  y1, z0)
    b_tr = v(x_back,  y1, z1)
    b_tl = v(x_back,  y0, z1)

    # front face (x = x_front)
    f_bl = v(x_front, y0, z0)
    f_br = v(x_front, y1, z0)
    f_tr = v(x_front, y1, z1)
    f_tl = v(x_front, y0, z1)

    bm.verts.ensure_lookup_table()

    def add_face(verts):
        try:
            return bm.faces.new(verts)
        except ValueError:
            return None

    # outward-facing quads
    add_face([b_bl, b_tl, b_tr, b_br])  # back
    add_face([f_bl, f_br, f_tr, f_tl])  # front
    add_face([b_bl, f_bl, f_tl, b_tl])  # y = y0 side
    add_face([b_br, b_tr, f_tr, f_br])  # y = y1 side
    add_face([b_bl, b_br, f_br, f_bl])  # z = z0 side
    add_face([b_tl, f_tl, f_tr, b_tr])  # z = z1 side

    bm.faces.ensure_lookup_table()
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj.data.materials.append(gray_mat)
    obj.data.materials.append(overlay_mat)

    # UV + material assignment
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    uv_layer = bm.loops.layers.uv.verify()

    def all_x_equal(face, x_target, eps=EPS):
        return all(abs(loop.vert.co.x - x_target) < eps for loop in face.loops)

    def set_uv_from_global_yz(face, flip_u=False):
        for loop in face.loops:
            co = loop.vert.co
            u = (co.y - y_outer_min) / (y_outer_max - y_outer_min)
            v = (co.z - z_outer_min) / (z_outer_max - z_outer_min)
            if flip_u:
                u = 1.0 - u
            loop[uv_layer].uv = (u, v)

    for f in bm.faces:
        f.material_index = 0

        # visible/front side
        if all_x_equal(f, x_front):
            f.material_index = 1
            set_uv_from_global_yz(f, flip_u=False)

        # back side, horizontally flipped
        elif all_x_equal(f, x_back):
            f.material_index = 1
            set_uv_from_global_yz(f, flip_u=True)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)

    return obj

# =========================================================
# CREATE THE 4 SOLID FRAME PIECES
# =========================================================
parts = []

# Top
parts.append(create_box_part(
    "gate_top",
    y_outer_min, y_outer_max,
    z_inner_max, z_outer_max
))

# Bottom
parts.append(create_box_part(
    "gate_bottom",
    y_outer_min, y_outer_max,
    z_outer_min, z_inner_min
))

# Left
parts.append(create_box_part(
    "gate_left",
    y_outer_min, y_inner_min,
    z_inner_min, z_inner_max
))

# Right
parts.append(create_box_part(
    "gate_right",
    y_inner_max, y_outer_max,
    z_inner_min, z_inner_max
))

# =========================================================
# EXPORT USD
# =========================================================
out_dir = os.path.dirname(USD_OUTPUT_PATH)
if out_dir:
    os.makedirs(out_dir, exist_ok=True)

bpy.ops.object.select_all(action='DESELECT')
root.select_set(True)
for p in parts:
    p.select_set(True)

bpy.context.view_layer.objects.active = root

bpy.ops.wm.usd_export(
    filepath=USD_OUTPUT_PATH,
    selected_objects_only=True
)

print("Created 4-part hollow gate asset")
print(f"Exported USD to: {USD_OUTPUT_PATH}")