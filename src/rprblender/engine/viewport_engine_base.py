#**********************************************************************
# Copyright 2020 Advanced Micro Devices, Inc
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#********************************************************************
import threading
import time
import math
from dataclasses import dataclass
import traceback
import textwrap

import bpy
import bgl
from gpu_extras.presets import draw_texture_2d
from bpy_extras import view3d_utils

import pyrpr
from .engine import Engine
from rprblender.export import camera, material, world, object, instance, particle
from rprblender.export.mesh import assign_materials
from rprblender.utils import gl
from rprblender import utils
from rprblender.utils.user_settings import get_user_settings

from rprblender.utils import logging
log = logging.Log(tag='viewport_engine')


MIN_ADAPT_RATIO_DIFF = 0.2
MIN_ADAPT_RESOLUTION_RATIO_DIFF = 0.1


@dataclass(init=False, eq=True)
class ViewportSettings:
    """
    Comparable dataclass which holds render settings for ViewportEngine:
    - camera viewport settings
    - render resolution
    - screen resolution
    - render border
    """

    camera_data: camera.CameraData
    screen_width: int
    screen_height: int
    border: tuple

    def __init__(self, context: bpy.types.Context):
        """Initializes settings from Blender's context"""
        self.camera_data = camera.CameraData.init_from_context(context)
        self.screen_width, self.screen_height = context.region.width, context.region.height

        scene = context.scene

        # getting render border
        x1, y1 = 0, 0
        x2, y2 = self.screen_width, self.screen_height
        if context.region_data.view_perspective == 'CAMERA':
            if scene.render.use_border:
                # getting border corners from camera view

                # getting screen camera points
                camera_obj = scene.camera
                camera_points = camera_obj.data.view_frame(scene=scene)
                screen_points = tuple(
                    view3d_utils.location_3d_to_region_2d(context.region,
                                                          context.space_data.region_3d,
                                                          camera_obj.matrix_world @ p)
                    for p in camera_points
                )

                # getting camera view region
                x1 = min(p[0] for p in screen_points)
                x2 = max(p[0] for p in screen_points)
                y1 = min(p[1] for p in screen_points)
                y2 = max(p[1] for p in screen_points)

                # adjusting region to border
                x, y = x1, y1
                dx, dy = x2 - x1, y2 - y1
                x1 = int(x + scene.render.border_min_x * dx)
                x2 = int(x + scene.render.border_max_x * dx)
                y1 = int(y + scene.render.border_min_y * dy)
                y2 = int(y + scene.render.border_max_y * dy)

                # adjusting to region screen resolution
                x1 = max(min(x1, self.screen_width), 0)
                x2 = max(min(x2, self.screen_width), 0)
                y1 = max(min(y1, self.screen_height), 0)
                y2 = max(min(y2, self.screen_height), 0)

        else:
            if context.space_data.use_render_border:
                # getting border corners from viewport camera

                x, y = x1, y1
                dx, dy = x2 - x1, y2 - y1
                x1 = int(x + context.space_data.render_border_min_x * dx)
                x2 = int(x + context.space_data.render_border_max_x * dx)
                y1 = int(y + context.space_data.render_border_min_y * dy)
                y2 = int(y + context.space_data.render_border_max_y * dy)

        # getting render resolution and render border
        width, height = x2 - x1, y2 - y1
        self.border = (x1, y1), (width, height)

    def export_camera(self, rpr_camera):
        """Exports camera settings with render border"""
        self.camera_data.export(rpr_camera,
            ((self.border[0][0] / self.screen_width, self.border[0][1] / self.screen_height),
             (self.border[1][0] / self.screen_width, self.border[1][1] / self.screen_height)))

    @property
    def width(self):
        return self.border[1][0]

    @property
    def height(self):
        return self.border[1][1]


@dataclass(init=False, eq=True)
class ShadingData:
    type: str
    use_scene_lights: bool = True
    use_scene_world: bool = True
    studio_light: str = None
    studio_light_rotate_z: float = 0.0
    studio_light_background_alpha: float = 0.0
    studio_light_intensity: float = 1.0

    def __init__(self, context: bpy.types.Context):
        shading = context.area.spaces.active.shading

        self.type = shading.type
        if self.type == 'RENDERED':
            self.use_scene_lights = shading.use_scene_lights_render
            self.use_scene_world = shading.use_scene_world_render
        else:
            self.use_scene_lights = shading.use_scene_lights
            self.use_scene_world = shading.use_scene_world

        if not self.use_scene_world:
            self.studio_light = shading.selected_studio_light.path
            if not self.studio_light:
                self.studio_light = str(utils.blender_data_dir() /
                                        "studiolights/world" / shading.studio_light)
            self.studio_light_rotate_z = shading.studiolight_rotate_z
            self.studio_light_background_alpha = shading.studiolight_background_alpha
            if hasattr(shading, "studiolight_intensity"):  # parameter added in Blender 2.81
                self.studio_light_intensity = shading.studiolight_intensity


@dataclass(init=False, eq=True)
class ViewLayerSettings:
    """
    Comparable dataclass which holds active view layer settings for ViewportEngine:
    - override material
    """

    material_override: bpy.types.Material = None

    def __init__(self, view_layer: bpy.types.ViewLayer):
        self.material_override = view_layer.material_override


def draw_texture(texture_id, x, y, width, height):
    # INITIALIZATION

    # Getting shader program
    shader_program = bgl.Buffer(bgl.GL_INT, 1)
    bgl.glGetIntegerv(bgl.GL_CURRENT_PROGRAM, shader_program)

    # Generate vertex array
    vertex_array = bgl.Buffer(bgl.GL_INT, 1)
    bgl.glGenVertexArrays(1, vertex_array)

    texturecoord_location = bgl.glGetAttribLocation(shader_program[0], "texCoord")
    position_location = bgl.glGetAttribLocation(shader_program[0], "pos")

    # Generate geometry buffers for drawing textured quad
    position = [x, y, x + width, y, x + width, y + height, x, y + height]
    position = bgl.Buffer(bgl.GL_FLOAT, len(position), position)
    texcoord = [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    texcoord = bgl.Buffer(bgl.GL_FLOAT, len(texcoord), texcoord)

    vertex_buffer = bgl.Buffer(bgl.GL_INT, 2)
    bgl.glGenBuffers(2, vertex_buffer)
    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
    bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, position, bgl.GL_STATIC_DRAW)
    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
    bgl.glBufferData(bgl.GL_ARRAY_BUFFER, 32, texcoord, bgl.GL_STATIC_DRAW)
    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

    # DRAWING
    bgl.glActiveTexture(bgl.GL_TEXTURE0)
    bgl.glBindTexture(bgl.GL_TEXTURE_2D, texture_id)

    bgl.glBindVertexArray(vertex_array[0])
    bgl.glEnableVertexAttribArray(texturecoord_location)
    bgl.glEnableVertexAttribArray(position_location)

    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[0])
    bgl.glVertexAttribPointer(position_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, vertex_buffer[1])
    bgl.glVertexAttribPointer(texturecoord_location, 2, bgl.GL_FLOAT, bgl.GL_FALSE, 0, None)
    bgl.glBindBuffer(bgl.GL_ARRAY_BUFFER, 0)

    bgl.glDrawArrays(bgl.GL_TRIANGLE_FAN, 0, 4)

    bgl.glBindVertexArray(0)
    bgl.glBindTexture(bgl.GL_TEXTURE_2D, 0)

    # DELETING
    bgl.glDeleteBuffers(2, vertex_buffer)
    bgl.glDeleteVertexArrays(1, vertex_array)


class ViewportEngineBase(Engine):
    """ Viewport render engine """

    TYPE = 'VIEWPORT'

    def __init__(self, rpr_engine):
        super().__init__(rpr_engine)

        self.shading_data = None
        self.space_data = None
        self.user_settings = get_user_settings()

    def stop_render(self):
        pass

    def sync(self, context, depsgraph):
        self.shading_data = ShadingData(context)
        self.space_data = context.space_data

    def sync_update(self, context, depsgraph):
        pass

    def draw(self, context):
        pass

    def depsgraph_objects(self, depsgraph, with_camera=False):
        for obj in super().depsgraph_objects(depsgraph, with_camera):
            if obj.type == 'LIGHT' and not self.shading_data.use_scene_lights:
                continue

            # check for local view visability
            if not obj.visible_in_viewport_get(self.space_data):
                continue

            yield obj

    def depsgraph_instances(self, depsgraph):
        for instance in super().depsgraph_instances(depsgraph):
            # check for local view visability
            if not instance.parent.visible_in_viewport_get(self.space_data):
                continue

            yield instance

    def _get_world_settings(self, depsgraph):
        if self.shading_data.use_scene_world:
            return world.WorldData.init_from_world(depsgraph.scene.world)

        return world.WorldData.init_from_shading_data(self.shading_data)
