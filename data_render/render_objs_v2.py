"""Blender script to render images of 3D models."""

import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, Generator, List, Literal, Optional, Set, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
import bpy_extras
import shutil

IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.import_scene.obj,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.import_mesh.ply,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}


def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def sample_point_on_sphere(radius: float, azimuth: float = None) -> Tuple[float, float, float]:
    """Samples a point on a sphere with the given radius.

    Args:
        radius (float): Radius of the sphere.

    Returns:
        Tuple[float, float, float]: A point on the sphere.
    """
    theta = random.random() * 2 * math.pi if azimuth is None else azimuth
    phi = None
    while phi is None or phi < np.pi / 4 or phi > np.pi / 2:
        phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def _sample_spherical(
    radius_min: float = 1.5,
    radius_max: float = 2.0,
    maxz: float = 1.6,
    minz: float = -0.75,
) -> np.ndarray:
    """Sample a random point in a spherical shell.

    Args:
        radius_min (float): Minimum radius of the spherical shell.
        radius_max (float): Maximum radius of the spherical shell.
        maxz (float): Maximum z value of the spherical shell.
        minz (float): Minimum z value of the spherical shell.

    Returns:
        np.ndarray: A random (x, y, z) point in the spherical shell.
    """
    correct = False
    vec = np.array([0, 0, 0])
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera(
    radius_min: float = 1.5,
    radius_max: float = 2.2,
    maxz: float = 1.2,
    minz: float = -1.2,
    only_northern_hemisphere: bool = False,
    azimuth: float = None,
) -> bpy.types.Object:
    """Randomizes the camera location and rotation inside of a spherical shell.

    Args:
        radius_min (float, optional): Minimum radius of the spherical shell. Defaults to
            1.5.
        radius_max (float, optional): Maximum radius of the spherical shell. Defaults to
            2.0.
        maxz (float, optional): Maximum z value of the spherical shell. Defaults to 1.6.
        minz (float, optional): Minimum z value of the spherical shell. Defaults to
            -0.75.
        only_northern_hemisphere (bool, optional): Whether to only sample points in the
            northern hemisphere. Defaults to False.

    Returns:
        bpy.types.Object: The camera object.
    """
    if azimuth is not None:
        radius = np.random.uniform(radius_min, radius_max, 1)
        x, y, z = sample_point_on_sphere(radius=radius[0], azimuth=azimuth)
    else:    
        x, y, z = _sample_spherical(
            radius_min=radius_min, radius_max=radius_max, maxz=maxz, minz=minz
        )
    camera = bpy.data.objects["Camera"]

    # only positive z
    if only_northern_hemisphere:
        z = abs(z)

    camera.location = Vector(np.array([x, y, z]))

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()

    return camera


def _set_camera_at_size(i: int, scale: float = 1.5) -> bpy.types.Object:
    """Debugging function to set the camera on the 6 faces of a cube.

    Args:
        i (int): Index of the face of the cube.
        scale (float, optional): Scale of the cube. Defaults to 1.5.

    Returns:
        bpy.types.Object: The camera object.
    """
    if i == 0:
        x, y, z = scale, 0, 0
    elif i == 1:
        x, y, z = -scale, 0, 0
    elif i == 2:
        x, y, z = 0, scale, 0
    elif i == 3:
        x, y, z = 0, -scale, 0
    elif i == 4:
        x, y, z = 0, 0, scale
    elif i == 5:
        x, y, z = 0, 0, -scale
    else:
        raise ValueError(f"Invalid index: i={i}, must be int in range [0, 5].")
    camera = bpy.data.objects["Camera"]
    camera.location = Vector(np.array([x, y, z]))
    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def _create_light(
    name: str,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"],
    location: Tuple[float, float, float],
    rotation: Tuple[float, float, float],
    energy: float,
    use_shadow: bool = False,
    specular_factor: float = 1.0,
):
    """Creates a light object.

    Args:
        name (str): Name of the light object.
        light_type (Literal["POINT", "SUN", "SPOT", "AREA"]): Type of the light.
        location (Tuple[float, float, float]): Location of the light.
        rotation (Tuple[float, float, float]): Rotation of the light.
        energy (float): Energy of the light.
        use_shadow (bool, optional): Whether to use shadows. Defaults to False.
        specular_factor (float, optional): Specular factor of the light. Defaults to 1.0.

    Returns:
        bpy.types.Object: The light object.
    """

    light_data = bpy.data.lights.new(name=name, type=light_type)
    light_object = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light_object)
    light_object.location = location
    light_object.rotation_euler = rotation
    light_data.use_shadow = use_shadow
    light_data.specular_factor = specular_factor
    light_data.energy = energy
    return light_object


def randomize_lighting() -> Dict[str, bpy.types.Object]:
    """Randomizes the lighting in the scene.

    Returns:
        Dict[str, bpy.types.Object]: Dictionary of the lights in the scene. The keys are
            "key_light", "fill_light", "rim_light", and "bottom_light".
    """

    # Clear existing lights
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    # Create key light
    key_light = _create_light(
        name="Key_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, -0.785398),
        energy=random.choice([3, 4, 5]),
    )

    # Create fill light
    fill_light = _create_light(
        name="Fill_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(0.785398, 0, 2.35619),
        energy=random.choice([2, 3, 4]),
    )

    # Create rim light
    rim_light = _create_light(
        name="Rim_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(-0.785398, 0, -3.92699),
        energy=random.choice([3, 4, 5]),
    )

    # Create bottom light
    bottom_light = _create_light(
        name="Bottom_Light",
        light_type="SUN",
        location=(0, 0, 0),
        rotation=(3.14159, 0, 0),
        energy=random.choice([1, 2, 3]),
    )

    return dict(
        key_light=key_light,
        fill_light=fill_light,
        rim_light=rim_light,
        bottom_light=bottom_light,
    )


def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)


def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def get_3x4_RT_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 3x4 RT matrix from the given camera.

    Taken from Zero123, which in turn was taken from
    https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py

    Args:
        cam (bpy.types.Object): The camera object.

    Returns:
        Matrix: The 3x4 RT matrix from the given camera.
    """
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT

def angle_between_obj_axis_and_xy_plane(obj, idx):
    z_axis = obj.matrix_world.to_3x3().col[idx]  # 获取物体的Z轴向量
    xy_plane_normal = Vector((0, 0, 1))  # X、Y平面的法向量（0, 0, 1）

    # 计算两个向量之间的夹角（弧度）
    angle_rad = math.acos(z_axis.dot(xy_plane_normal) / (z_axis.length * xy_plane_normal.length))

    # 将弧度转换为角度
    angle_deg = math.degrees(angle_rad)

    return angle_deg


def get_4x4_perspective_matrix_from_blender(cam: bpy.types.Object) -> Matrix:
    """Returns the 4x4 perspective matrix from the given camera."""
    # Access the camera data
    cam_data = cam.data

    # Aspect ratio
    scene = bpy.context.scene
    render = scene.render
    aspect_ratio = render.resolution_x / render.resolution_y

    # Field of View (in radians)
    fovy = cam_data.angle
    fovx = 2 * np.arctan(np.tan(fovy / 2) * aspect_ratio)
    
    # Near and Far clipping planes
    near = cam_data.clip_start
    far = cam_data.clip_end
    
    # Perspective matrix components
    f = 1 / np.tan(fovy / 2)
    matrix = Matrix((
        (f / aspect_ratio, 0, 0, 0),
        (0, f, 0, 0),
        (0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)),
        (0, 0, -1, 0)
    ))

    return cam.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )
    
    return matrix

def add_lighting():
    # 设置HDRI背景
    hdr_image_path = "/data/longbinji/blender-3.2.2-linux-x64/thatch_chapel_4k.hdr"
    
    if bpy.context.scene.world is None:
        new_world = bpy.data.worlds.new("NewWorld")
        bpy.context.scene.world = new_world
    
    bpy.context.scene.world.use_nodes = True
    world_nodes = bpy.context.scene.world.node_tree.nodes
    world_links = bpy.context.scene.world.node_tree.links

    # 清除现有节点
    for node in world_nodes:
        world_nodes.remove(node)

    # 创建新的节点
    background_node = world_nodes.new(type='ShaderNodeBackground')
    environment_texture_node = world_nodes.new(type='ShaderNodeTexEnvironment')
    environment_texture_node.image = bpy.data.images.load(hdr_image_path)
    output_node = world_nodes.new(type='ShaderNodeOutputWorld')

    # 连接节点
    world_links.new(environment_texture_node.outputs['Color'], background_node.inputs['Color'])
    world_links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    

    bpy.ops.object.light_add(type='SUN', location=(0, 10, 1))
    sun = bpy.context.object
    sun.data.energy = 3

def add_floor():
    texture_image_path = "/data/longbinji/blender-3.2.2-linux-x64/floor_tiles_06_4k.blend/textures/floor_tiles_06_diff_4k.jpg"
    
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    
    tile_count = 3
    tile_size = 10

    # 创建新的材质并添加到地板
    material = bpy.data.materials.new(name="FloorMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # 添加纹理节点并设置路径
    tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_image_path)
    material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # 将材质分配给地板
    if floor.data.materials:
        floor.data.materials[0] = material
    else:
        floor.data.materials.append(material)
        
    for i in range(-tile_count, tile_count):
        for j in range(-tile_count, tile_count):
            if i == 0 and j == 0:
                continue  # 跳过原始地板
            new_floor = floor.copy()
            new_floor.data = floor.data.copy()
            new_floor.location = (i * tile_size, j * tile_size, 0)
            bpy.context.collection.objects.link(new_floor)


def delete_invisible_objects() -> None:
    """Deletes all invisible objects in the scene.

    Returns:
        None
    """
    bpy.ops.object.select_all(action="DESELECT")
    for obj in scene.objects:
        if obj.hide_viewport or obj.hide_render:
            obj.hide_viewport = False
            obj.hide_render = False
            obj.hide_select = False
            obj.select_set(True)
    bpy.ops.object.delete()

    # Delete invisible collections
    invisible_collections = [col for col in bpy.data.collections if col.hide_viewport]
    for col in invisible_collections:
        bpy.data.collections.remove(col)


def normalize_scene() -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.

    Mostly taken from the Point-E / Shap-E rendering script
    (https://github.com/openai/point-e/blob/main/point_e/evals/scripts/blender_script.py#L97-L112),
    but fix for multiple root objects: (see bug report here:
    https://github.com/openai/shap-e/pull/60).

    Returns:
        None
    """
    if len(list(get_scene_root_objects())) > 1:
        # create an empty object to be used as a parent for all root objects
        parent_empty = bpy.data.objects.new("ParentEmpty", None)
        bpy.context.scene.collection.objects.link(parent_empty)

        # parent all root objects to the empty object
        for obj in get_scene_root_objects():
            if obj != parent_empty:
                obj.parent = parent_empty

    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in get_scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in get_scene_root_objects():
        print(obj.name)
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

    # unparent the camera
    bpy.data.objects["Camera"].parent = None


def has_global_motion(thresh) -> bool:
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    return max(offset) > thresh


def has_large_scale_change(thresh) -> bool:
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    return scale > thresh


def delete_missing_textures() -> Dict[str, Any]:
    """Deletes all missing textures in the scene.

    Returns:
        Dict[str, Any]: Dictionary with keys "count", "files", and "file_path_to_color".
            "count" is the number of missing textures, "files" is a list of the missing
            texture file paths, and "file_path_to_color" is a dictionary mapping the
            missing texture file paths to a random color.
    """
    missing_file_count = 0
    out_files = []
    file_path_to_color = {}

    # Check all materials in the scene
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == "TEX_IMAGE":
                    image = node.image
                    if image is not None:
                        file_path = bpy.path.abspath(image.filepath)
                        if file_path == "":
                            # means it's embedded
                            continue

                        if not os.path.exists(file_path):
                            # Find the connected Principled BSDF node
                            connected_node = node.outputs[0].links[0].to_node

                            if connected_node.type == "BSDF_PRINCIPLED":
                                if file_path not in file_path_to_color:
                                    # Set a random color for the unique missing file path
                                    random_color = [random.random() for _ in range(3)]
                                    file_path_to_color[file_path] = random_color + [1]

                                connected_node.inputs[
                                    "Base Color"
                                ].default_value = file_path_to_color[file_path]

                            # Delete the TEX_IMAGE node
                            material.node_tree.nodes.remove(node)
                            missing_file_count += 1
                            out_files.append(image.filepath)
    return {
        "count": missing_file_count,
        "files": out_files,
        "file_path_to_color": file_path_to_color,
    }


def _get_random_color() -> Tuple[float, float, float, float]:
    """Generates a random RGB-A color.

    The alpha value is always 1.

    Returns:
        Tuple[float, float, float, float]: A random RGB-A color. Each value is in the
        range [0, 1].
    """
    return (random.random(), random.random(), random.random(), 1)


def _apply_color_to_object(
    obj: bpy.types.Object, color: Tuple[float, float, float, float]
) -> None:
    """Applies the given color to the object.

    Args:
        obj (bpy.types.Object): The object to apply the color to.
        color (Tuple[float, float, float, float]): The color to apply to the object.

    Returns:
        None
    """
    mat = bpy.data.materials.new(name=f"RandomMaterial_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")
    if principled_bsdf:
        principled_bsdf.inputs["Base Color"].default_value = color
    obj.data.materials.append(mat)


def apply_single_random_color_to_all_objects() -> Tuple[float, float, float, float]:
    """Applies a single random color to all objects in the scene.

    Returns:
        Tuple[float, float, float, float]: The random color that was applied to all
        objects.
    """
    rand_color = _get_random_color()
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_color_to_object(obj, rand_color)
    return rand_color


class MetadataExtractor:
    """Class to extract metadata from a Blender scene."""

    def __init__(
        self, object_path: str, scene: bpy.types.Scene, bdata: bpy.types.BlendData
    ) -> None:
        """Initializes the MetadataExtractor.

        Args:
            object_path (str): Path to the object file.
            scene (bpy.types.Scene): The current scene object from `bpy.context.scene`.
            bdata (bpy.types.BlendData): The current blender data from `bpy.data`.

        Returns:
            None
        """
        self.object_path = object_path
        self.scene = scene
        self.bdata = bdata

    def get_poly_count(self) -> int:
        """Returns the total number of polygons in the scene."""
        total_poly_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_poly_count += len(obj.data.polygons)
        return total_poly_count

    def get_vertex_count(self) -> int:
        """Returns the total number of vertices in the scene."""
        total_vertex_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_vertex_count += len(obj.data.vertices)
        return total_vertex_count

    def get_edge_count(self) -> int:
        """Returns the total number of edges in the scene."""
        total_edge_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                total_edge_count += len(obj.data.edges)
        return total_edge_count

    def get_lamp_count(self) -> int:
        """Returns the number of lamps in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "LIGHT")

    def get_mesh_count(self) -> int:
        """Returns the number of meshes in the scene."""
        return sum(1 for obj in self.scene.objects if obj.type == "MESH")

    def get_material_count(self) -> int:
        """Returns the number of materials in the scene."""
        return len(self.bdata.materials)

    def get_object_count(self) -> int:
        """Returns the number of objects in the scene."""
        return len(self.bdata.objects)

    def get_animation_count(self) -> int:
        """Returns the number of animations in the scene."""
        return len(self.bdata.actions)

    def get_linked_files(self) -> List[str]:
        """Returns the filepaths of all linked files."""
        image_filepaths = self._get_image_filepaths()
        material_filepaths = self._get_material_filepaths()
        linked_libraries_filepaths = self._get_linked_libraries_filepaths()

        all_filepaths = (
            image_filepaths | material_filepaths | linked_libraries_filepaths
        )
        if "" in all_filepaths:
            all_filepaths.remove("")
        return list(all_filepaths)

    def _get_image_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in the scene."""
        filepaths = set()
        for image in self.bdata.images:
            if image.source == "FILE":
                filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_material_filepaths(self) -> Set[str]:
        """Returns the filepaths of all images used in materials."""
        filepaths = set()
        for material in self.bdata.materials:
            if material.use_nodes:
                for node in material.node_tree.nodes:
                    if node.type == "TEX_IMAGE":
                        image = node.image
                        if image is not None:
                            filepaths.add(bpy.path.abspath(image.filepath))
        return filepaths

    def _get_linked_libraries_filepaths(self) -> Set[str]:
        """Returns the filepaths of all linked libraries."""
        filepaths = set()
        for library in self.bdata.libraries:
            filepaths.add(bpy.path.abspath(library.filepath))
        return filepaths

    def get_scene_size(self) -> Dict[str, list]:
        """Returns the size of the scene bounds in meters."""
        bbox_min, bbox_max = scene_bbox()
        return {"bbox_max": list(bbox_max), "bbox_min": list(bbox_min)}

    def get_shape_key_count(self) -> int:
        """Returns the number of shape keys in the scene."""
        total_shape_key_count = 0
        for obj in self.scene.objects:
            if obj.type == "MESH":
                shape_keys = obj.data.shape_keys
                if shape_keys is not None:
                    total_shape_key_count += (
                        len(shape_keys.key_blocks) - 1
                    )  # Subtract 1 to exclude the Basis shape key
        return total_shape_key_count

    def get_armature_count(self) -> int:
        """Returns the number of armatures in the scene."""
        total_armature_count = 0
        for obj in self.scene.objects:
            if obj.type == "ARMATURE":
                total_armature_count += 1
        return total_armature_count

    def read_file_size(self) -> int:
        """Returns the size of the file in bytes."""
        return os.path.getsize(self.object_path)

    def get_metadata(self) -> Dict[str, Any]:
        """Returns the metadata of the scene.

        Returns:
            Dict[str, Any]: Dictionary of the metadata with keys for "file_size",
            "poly_count", "vert_count", "edge_count", "material_count", "object_count",
            "lamp_count", "mesh_count", "animation_count", "linked_files", "scene_size",
            "shape_key_count", and "armature_count".
        """
        return {
            "file_size": self.read_file_size(),
            "poly_count": self.get_poly_count(),
            "vert_count": self.get_vertex_count(),
            "edge_count": self.get_edge_count(),
            "material_count": self.get_material_count(),
            "object_count": self.get_object_count(),
            "lamp_count": self.get_lamp_count(),
            "mesh_count": self.get_mesh_count(),
            "animation_count": self.get_animation_count(),
            "linked_files": self.get_linked_files(),
            "scene_size": self.get_scene_size(),
            "shape_key_count": self.get_shape_key_count(),
            "armature_count": self.get_armature_count(),
        }


def duplicate_and_apply_modifiers(frame):
    bpy.context.scene.frame_set(frame)
    bpy.ops.object.duplicate(linked=False, mode='TRANSLATION')
    duplicated_obj = bpy.context.active_object
    
    # Apply all Armature modifiers
    for modifier in duplicated_obj.modifiers:
        if modifier.type == 'ARMATURE':
            bpy.ops.object.modifier_apply(modifier=modifier.name)
    
    return duplicated_obj


def render_object(
    object_file: str,
    num_renders: int,
    only_northern_hemisphere: bool,
    output_dir: str,
    render_animation: bool = False,
    max_n_frames: int = None,
    render: bool = True,
    export_mesh: bool = False,
    filter_object_with_global_motion: bool = False,
    global_motion_threshold: float = 0.5,
    filter_object_with_large_scale_change: bool = False,
    large_scale_change_threshold: float = 0.5,
    uniform_azimuth: bool = False,
    needed_actions: List[str] = None,
    missing_file: str = None,
    save_idx_num: int = 0,
    
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    curve_types = ["S", "circle"]
    curve_type = curve_types[random.randint(0, 1)]
    
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.split(object_file)[-1][:-4]
    out_video_path = os.path.join(output_dir, f"{basename}_{save_idx_num}_{curve_type}")
    out_video_path_depth = os.path.join(output_dir, f"{basename}_{save_idx_num}_{curve_type}", "depth")

    if not os.path.exists(out_video_path):
        os.makedirs(out_video_path)
        
    if not os.path.exists(out_video_path_depth):
        os.makedirs(out_video_path_depth)
    
    if not render and not export_mesh:
        raise ValueError("At least one of render or export_mesh must be True.")
    if not render:
        num_renders = 1

    # load the object
    if object_file.endswith(".blend"):
        bpy.ops.object.mode_set(mode="OBJECT")
        reset_cameras()
        delete_invisible_objects()
    else:
        reset_scene()
        reset_cameras()
        load_object(object_file)

    # Set up cameras
    cam = scene.objects["Camera"]
#    cam.data.lens = 35
#    cam.data.sensor_width = 32
    cam.location = (0, -4.7, 2.3)
    cam.rotation_euler = (math.radians(67), 0, 0)
    cam.data.clip_start = 0.1

    # Set up camera constraints
    # cam_constraint = cam.constraints.new(type="TRACK_TO")
    # cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    # cam_constraint.up_axis = "UP_Y"
    # empty = bpy.data.objects.new("Empty", None)
    # scene.collection.objects.link(empty)
    # cam_constraint.target = empty

    # Extract the metadata. This must be done before normalizing the scene to get
    # accurate bounding box information.
    metadata_extractor = MetadataExtractor(
        object_path=object_file, scene=scene, bdata=bpy.data
    )
    metadata = metadata_extractor.get_metadata()

    # delete all objects that are not meshes
    if object_file.lower().endswith(".usdz"):
        # don't delete missing textures on usdz files, lots of them are embedded
        missing_textures = None
    else:
        # missing_textures = delete_missing_textures()
        missing_textures = None
    metadata["missing_textures"] = missing_textures

    # possibly apply a random color to all objects
    if object_file.endswith(".stl") or object_file.endswith(".ply"):
        assert len(bpy.context.selected_objects) == 1
        rand_color = apply_single_random_color_to_all_objects()
        metadata["random_color"] = rand_color
    else:
        metadata["random_color"] = None

    # save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, sort_keys=True, indent=2)

    # normalize the scene
    normalize_scene()
    
    parent_obj = bpy.data.objects["ParentEmpty"]
    
    lowest_point = float('inf')
    min_coord = Vector((float('inf'), float('inf'), float('inf')))
    max_coord = Vector((float('-inf'), float('-inf'), float('-inf')))

    for obj in bpy.data.objects:
            # 检查子对象是否为网格
        if obj.type == 'MESH':
            for vert in obj.data.vertices:
                # 将顶点位置转换为世界坐标
                world_location = obj.matrix_world @ vert.co
                # 更新最低点的坐标
                if world_location.z < lowest_point:
                    lowest_point = world_location.z

                min_coord = Vector((min(a, b) for a, b in zip(min_coord, world_location)))
                max_coord = Vector((max(a, b) for a, b in zip(max_coord, world_location)))
    dimensions = max_coord - min_coord
    
    new_height = dimensions.z
    new_width = dimensions.x
    new_depth = dimensions.y
    print(dimensions, new_height/new_width)
    
    if new_height/new_width < 0.1:
        with open(missing_file, 'a') as file:
            file.write(f'{object_file}\n')
        return
    
    parent_obj.location.z -= lowest_point
    
    min_point = min_coord
    max_point = max_coord
    
    new_location = Vector((0, 0, 0))
    
    bpy.ops.object.empty_add(location=new_location)
    new_parent = bpy.context.object
    
    parent_obj.parent = new_parent
    parent_obj = new_parent
    
    texture_image_path = "/data/longbinji/blender-3.2.2-linux-x64/floor_tiles_06_4k.blend/textures/floor_tiles_06_diff_4k.jpg"
    
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    floor = bpy.context.object
    floor.name = "Floor"
    
    tile_count = 2
    tile_size = 10

    # 创建新的材质并添加到地板
    material = bpy.data.materials.new(name="FloorMaterial")
    material.use_nodes = True
    bsdf = material.node_tree.nodes["Principled BSDF"]

    # 添加纹理节点并设置路径
    tex_image = material.node_tree.nodes.new('ShaderNodeTexImage')
    tex_image.image = bpy.data.images.load(texture_image_path)
    material.node_tree.links.new(bsdf.inputs['Base Color'], tex_image.outputs['Color'])

    # 将材质分配给地板
    if floor.data.materials:
        floor.data.materials[0] = material
    else:
        floor.data.materials.append(material)
        
    for i in range(-tile_count, tile_count):
        for j in range(-tile_count, tile_count):
            if i == 0 and j == 0:
                continue  # 跳过原始地板
            new_floor = floor.copy()
            new_floor.data = floor.data.copy()
            new_floor.location = (i * tile_size, j * tile_size, 0)
            bpy.context.collection.objects.link(new_floor)
    add_lighting()
    
    # curve_type = curve_types[0]
    random_angle = random.randint(0, 90)
    radius = 1.1
    curve_data = bpy.data.curves.new('Trajectory', type='CURVE')
    curve_data.dimensions = '3D'
    polyline = curve_data.splines.new('POLY')
    num_points = 32
    polyline.points.add(num_points-1)

    if curve_type == "circle":
        for i in range(num_points):
            angle = math.pi * i / (num_points - 1) + random_angle
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            polyline.points[i].co = (x, y, 0, 1)
    elif curve_type == "tuo":
        for i in range(num_points):
            angle = math.pi * i / (num_points - 1) + random_angle
            x = radius * math.cos(angle)
            y = radius * 0.5 * math.sin(angle)
            polyline.points[i].co = (x, y, 0, 1)
    elif curve_type == "S":
        radius=0.49
        for i in range(num_points // 2):
            angle = math.pi * i / (num_points // 2 - 1) + random_angle
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            polyline.points[i].co = (x, y, 0, 1)
            
        x_moved = 2*radius*math.cos(random_angle)
        y_moved = 2*radius*math.sin(random_angle)

        # 反向半圆
        for i in range(num_points // 2):
            angle = math.pi * i / (num_points // 2 - 1) + random_angle
            x = radius * math.cos(angle)
            y = -radius * math.sin(angle)
            polyline.points[(num_points-i-1)].co = (-x-x_moved, y - y_moved, 0, 1)
    elif curve_type == "line":
        length = 2*radius
        for i in range(num_points):
            x = math.cos(random_angle)*(length/(num_points-1))*i
            y = math.sin(random_angle)*(length/(num_points-1))*i
            polyline.points[i].co = (x, y, 0, 1)
    
    curve_object = bpy.data.objects.new('Trajectory', curve_data)
    bpy.context.collection.objects.link(curve_object)
    
    pos = np.zeros((len(polyline.points), 2))
    start_box = np.zeros((len(polyline.points), 8, 2))
    depth = np.zeros((len(polyline.points), 3))
    
    for idx, save_point in enumerate(polyline.points):
        point_world = Vector((save_point.co))  # 替换为你的3D点的世界坐标
        camera_matrix_world = cam.matrix_world
        camera_matrix_world_inv = camera_matrix_world.inverted()
        point_local = camera_matrix_world_inv @ point_world
        loc_x = point_local.x
        loc_y = point_local.y
        loc_z = point_local.z
        # 获取相机的投影矩阵
        scale = render.resolution_percentage / 100.0
        width = int(render.resolution_x * scale)
        height = int(render.resolution_y * scale)

        # 获取相机数据
        cam_data = cam.data

        # 使用Blender的API将局部坐标转换为图像坐标
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, point_world)

        # 转换为像素坐标
        x = int(co_2d.x * width)
        y = int((1 - co_2d.y) * height)  # Blender的Y坐标是从底部开始的，所以需要 (1 - co_2d.y)
               
        pos[idx] = [x,y]
        depth[idx] = [loc_x,loc_y,loc_z]
        print(depth[idx])
#        print(x,y)
        
        if curve_type == "line":
            rot_width = (new_width/2)
            rot_depth = (new_depth/2)
            
        else:
            rot_x = save_point.co[0]
            rot_y = save_point.co[1]
            angle_rot = math.atan(rot_x/(rot_y+0.0000000001))
            angle_rot2 = math.atan(rot_y/(rot_x+0.0000000001))
            ori_angle = math.atan(new_depth/(new_width+0.0000000001))
            ori_angle2 = math.atan(new_width/(new_depth+0.0000000001))
            
            edge = new_depth/math.sin(ori_angle)
            rot_width = (edge/2)*math.sin(angle_rot+ori_angle2)
            rot_depth = (edge/2)*math.cos(angle_rot+ori_angle2)
            rot_width_2 = (edge/2)*math.cos(angle_rot2+ori_angle2)
            rot_depth_2 = (edge/2)*math.sin(angle_rot2+ori_angle2)

        
#        init_point_1 = mathutils.Vector((save_point.co)) + mathutils.Vector((-rot_width, -rot_depth, 0, 0))
        init_point_1 = Vector((save_point.co)) + Vector((-rot_width, -rot_depth, 0, 0))
        init_point_2 = Vector((save_point.co)) + Vector((rot_width, rot_depth, 0, 0))
        
        init_point_3 = Vector((save_point.co)) + Vector((-rot_width_2, -rot_depth_2, 0, 0))
        init_point_4 = Vector((save_point.co)) + Vector((rot_width_2, rot_depth_2, 0, 0))
        
        init_point_5 = Vector((save_point.co)) + Vector((-rot_width, -rot_depth, new_height, 0))
        init_point_6 = Vector((save_point.co)) + Vector((rot_width, rot_depth, new_height, 0))
        
        init_point_7 = Vector((save_point.co)) + Vector((-rot_width_2, -rot_depth_2, new_height, 0))
        init_point_8 = Vector((save_point.co)) + Vector((rot_width_2, rot_depth_2, new_height, 0))
        
        co_2d_1 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_1)
        co_2d_2 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_2)
        co_2d_3 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_3)
        co_2d_4 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_4)
        co_2d_5 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_5)
        co_2d_6 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_6)
        co_2d_7 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_7)
        co_2d_8 = bpy_extras.object_utils.world_to_camera_view(scene, cam, init_point_8)

        init_1_x = int(co_2d_1.x * width)
        init_1_y = int((1 - co_2d_1.y) * height)  # Blender的Y坐标是从底部开始的，所以需要 (1 - co_2d.y)
        
        init_2_x = int(co_2d_2.x * width)
        init_2_y = int((1 - co_2d_2.y) * height)
        
        init_3_x = int(co_2d_3.x * width)
        init_3_y = int((1 - co_2d_3.y) * height)
        
        init_4_x = int(co_2d_4.x * width)
        init_4_y = int((1 - co_2d_4.y) * height)
        
        init_5_x = int(co_2d_5.x * width)
        init_5_y = int((1 - co_2d_5.y) * height)  # Blender的Y坐标是从底部开始的，所以需要 (1 - co_2d.y)
        
        init_6_x = int(co_2d_6.x * width)
        init_6_y = int((1 - co_2d_6.y) * height)
        
        init_7_x = int(co_2d_7.x * width)
        init_7_y = int((1 - co_2d_7.y) * height)
        
        init_8_x = int(co_2d_8.x * width)
        init_8_y = int((1 - co_2d_8.y) * height)
               
        start_box[idx][0] = [init_1_x, init_1_y]
        start_box[idx][1] = [init_2_x, init_2_y]
        start_box[idx][2] = [init_3_x, init_3_y]
        start_box[idx][3] = [init_4_x, init_4_y]
        start_box[idx][4] = [init_5_x, init_5_y]
        start_box[idx][5] = [init_6_x, init_6_y]
        start_box[idx][6] = [init_7_x, init_7_y]
        start_box[idx][7] = [init_8_x, init_8_y]
        
        print(start_box[idx])
    
    np.save(f"{out_video_path}/traj.npy", pos)
    np.save(f"{out_video_path}/bbox.npy", start_box)
    # np.save(f"{out_video_path}/{basename}_{save_idx_num}_{curve_type}_depth.npy", depth)
    
    # 让对象沿着轨迹运动
    bpy.context.view_layer.objects.active = parent_obj
    constraint = parent_obj.constraints.new(type='FOLLOW_PATH')
    constraint.target = curve_object
    constraint.use_fixed_location = True
    
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, parent_obj.location.z))
    target_empty = bpy.context.object
    target_empty.name = "Target"

    locked_track_constraint = parent_obj.constraints.new(type='DAMPED_TRACK')
    locked_track_constraint.target = target_empty
    constraint.use_fixed_location = True

    locked_track_constraint.track_axis = 'TRACK_NEGATIVE_Y'

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 32
    
    for i in range(num_points):
        bpy.context.scene.frame_set(i + 1)
        constraint.offset_factor = i / (num_points - 1)
        parent_obj.keyframe_insert(data_path='constraints["Follow Path"].offset_factor', frame=i + 1)
        
#    assert False

    # 设置输出路径
    output_path = out_video_path_depth
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)

    # 启用Z-pass
    bpy.context.view_layer.use_pass_z = True

    # 创建一个新的文件输出节点
    scene.use_nodes = True
    scene.node_tree.nodes.clear()
    
    render_layers = scene.node_tree.nodes.new(type='CompositorNodeRLayers')
    file_output_node = scene.node_tree.nodes.new(type='CompositorNodeOutputFile')

    # 设置文件输出节点的属性
    file_output_node.base_path = output_path
    file_output_node.format.file_format = 'OPEN_EXR'
    file_output_node.file_slots[0].path = "depth_"

    # 连接深度输出到文件输出节点
    scene.node_tree.links.new(render_layers.outputs['Depth'], file_output_node.inputs[0])


    bpy.context.scene.render.image_settings.file_format = 'FFMPEG'
    bpy.context.scene.render.ffmpeg.format = 'MPEG4'
    bpy.context.scene.render.filepath = out_video_path
    bpy.context.scene.render.fps = 5
    bpy.context.scene.render.resolution_x = 720
    bpy.context.scene.render.resolution_y = 480

    # 渲染动画
    bpy.ops.render.render(animation=True)
    
#    assert False
    
    
    
    bpy.data.objects.remove(floor)
    bpy.data.objects.remove(curve_object)
    for obj in bpy.data.objects:
        if obj.type in {'MESH', 'CURVE', 'LIGHT'}:
            bpy.data.objects.remove(obj)
    for material in bpy.data.materials:
        bpy.data.materials.remove(material)
    for image in bpy.data.images:
        bpy.data.images.remove(image)
        
#    assert False
                 
    return


# if __name__ == "__main__":
for obj in bpy.data.objects:
    if obj.type in {'MESH', 'CURVE', 'LIGHT'}:
        bpy.data.objects.remove(obj)
for material in bpy.data.materials:
    bpy.data.materials.remove(material)
for image in bpy.data.images:
    bpy.data.images.remove(image)
    

all_data_path = "/data/longbinji/hf-objaverse-v1-split2/glbs"

wrong_list = []

num = 0
total_num = 10000
cache_num = 0
object_start = 700
object_end = 800
sample_single = 5

output_dir = f"/data/longbinji/blender-3.2.2-linux-x64/render_videos_subset_split2_{object_start}_{object_end}"
rendered_file_src = "rendered_split.txt"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
missing_file = os.path.join(output_dir, "missing.txt")
rendered_file = os.path.join(output_dir, "rendered_split.txt")

if not os.path.exists(rendered_file):
    # assert 
    shutil.copy(rendered_file_src, rendered_file)

writed_file = os.path.join(output_dir, "writed.txt")

rendered_list = []
writed_list = []

with open(rendered_file, "r") as file:  
    for line in file:        
        stripped_line = line.strip()
        if "_" in  stripped_line:
            stripped_line = stripped_line.split("_")[0]
        if len(stripped_line)> 1:
            if stripped_line not in rendered_list:      
                rendered_list.append(stripped_line)

if os.path.exists(writed_file):         
    with open(writed_file, "r") as file:  
        for line in file:        
            stripped_line = line.strip()
            if len(stripped_line)> 1:      
                writed_list.append(stripped_line.split("-")[-1][1:])
                  
print(len(rendered_list))
 
for file in sorted(os.listdir(all_data_path)):
    for obj_file in os.listdir(os.path.join(all_data_path, file)):
        for i in range(sample_single):
            if obj_file.endswith("glb") and num<=total_num and (obj_file[:-4] in rendered_list[object_start:object_end]) and (obj_file[:-4] not in writed_list):
                basename = obj_file[:-4]
                if basename in wrong_list:
                    continue
                elif num < cache_num:
                    num += 1
                    print(f"Skip {obj_file}")
                    continue
                else:
                    print(basename)
                    # assert False
                
                    object_path = os.path.join(all_data_path, file, obj_file)
                    
                    output_dir = output_dir
                    engine = "CYCLES"
                    only_northern_hemisphere=True
                    num_renders=12
                    max_n_frames = 32
                    filter_object_with_global_motion=True
                    global_motion_threshold=0.3
                    filter_object_with_large_scale_change=True
                    large_scale_change_threshold=1.5
                    render=True
                    export_mesh=True
                    uniform_azimuth=True
                    actions=None
                    

                    context = bpy.context
                    scene = context.scene
                    render = scene.render

                    # Set render settings
                    render.engine = engine
                    render.image_settings.file_format = "PNG"
                    render.image_settings.color_mode = "RGBA"
                    render.resolution_x = 720
                    render.resolution_y = 480
                    render.resolution_percentage = 100

                    # Set cycles settings
                    scene.cycles.device = "GPU"
                    scene.cycles.samples = 128
                    scene.cycles.diffuse_bounces = 1
                    scene.cycles.glossy_bounces = 1
                    scene.cycles.transparent_max_bounces = 3
                    scene.cycles.transmission_bounces = 3
                    scene.cycles.filter_width = 0.01
                    scene.cycles.use_denoising = True
                    scene.render.film_transparent = True
                    # bpy.context.preferences.addons["cycles"].preferences.get_devices()
                    # bpy.context.preferences.addons[
                    #     "cycles"
                    # ].preferences.compute_device_type = "CUDA"  # or "OPENCL"
                    
                    device_index = 1  # Change this value to the index of your desired GPU
                    preferences = bpy.context.preferences.addons["cycles"].preferences
                    preferences.compute_device_type = "CUDA"
                    preferences.get_devices()
                    
                    for device in preferences.devices:
                        device.use = False
                    preferences.devices[device_index].use = True

                    # Render the images
                    all_excluded = render_object(
                        object_file=object_path,
                        num_renders=num_renders,
                        only_northern_hemisphere=only_northern_hemisphere,
                        output_dir=output_dir,
                        render_animation=True,
                        max_n_frames=max_n_frames,
                        render=render,
                        export_mesh=export_mesh,
                        filter_object_with_global_motion=filter_object_with_global_motion,
                        global_motion_threshold=global_motion_threshold,
                        filter_object_with_large_scale_change=filter_object_with_large_scale_change,
                        large_scale_change_threshold=large_scale_change_threshold,
                        uniform_azimuth=uniform_azimuth,
                        needed_actions=actions,
                        missing_file=missing_file,
                        save_idx_num=i,
                    )
                    if i == sample_single -1:
                        with open(writed_file, "a") as f:
                            f.write(f"{num} -- {basename}\n")
                    num += 1