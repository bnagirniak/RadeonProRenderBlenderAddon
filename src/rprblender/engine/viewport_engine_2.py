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
import time
import math

import pyrpr
import bpy

from rprblender.export import object
from .viewport_engine import (
    ViewportEngine, ViewportSettings, ShadingData, FinishRenderException
)
from rprblender.utils import gl
from .context import RPRContext2

from rprblender.utils import logging
log = logging.Log(tag='viewport_engine_2')


class ViewportEngine2(ViewportEngine):
    _RPRContext = RPRContext2

    def __init__(self, rpr_engine):
        super().__init__(rpr_engine)

        self.is_intermediate_render = False
        self.abort_render_iteration = False
        self.rendered_image = None

    def stop_render(self):
        self.abort_render_iteration = True
        super().stop_render()

    def _do_render(self):
        iteration = 0
        time_begin = 0.0
        time_render = 0.0

        update_iterations = 1

        def render_update(progress):
            # if iteration == 0:
            #     return

            if progress == 1.0:
                return

            if self.abort_render_iteration:
                self.abort_render_iteration = False
                self.rpr_context.abort_render()
                return

            # log("update_render_callback", progress)
            self._resolve()
            self.rendered_image = self.rpr_context.get_image()

            time_render = time.perf_counter() - time_begin
            it = iteration + int(update_iterations * progress)
            if self.render_iterations > 0:
                info_str = f"Time: {time_render:.1f} sec" \
                           f" | Iteration: {it}/{self.render_iterations}"
            else:
                info_str = f"Time: {time_render:.1f}/{self.render_time} sec" \
                           f" | Iteration: {it}"

            self.is_intermediate_render = True
            self.is_rendered = True
            self.notify_status(info_str, "Render")

        self.rpr_context.set_render_update_callback(render_update)

        self.notify_status("Starting...", "Render")

        is_adaptive = self.rpr_context.is_aov_enabled(pyrpr.AOV_VARIANCE)

        # Infinite cycle, which starts when scene has to be re-rendered.
        # It waits for restart_render_event be enabled.
        # Exit from this cycle is implemented through raising FinishRender
        # when self.is_finished be enabled from main thread.
        while True:
            self.restart_render_event.wait()

            if self.is_finished:
                raise FinishRenderException

            # preparations to start rendering
            iteration = 0
            time_begin = 0.0
            time_render = 0.0
            if is_adaptive:
                all_pixels = active_pixels = self.rpr_context.width * self.rpr_context.height
            is_last_iteration = False

            # this cycle renders each iteration
            while True:
                if self.is_finished:
                    raise FinishRenderException

                is_adaptive_active = is_adaptive and iteration >= self.rpr_context.get_parameter(
                    pyrpr.CONTEXT_ADAPTIVE_SAMPLING_MIN_SPP)

                if self.restart_render_event.is_set():
                    # clears restart_render_event, prepares to start rendering
                    self.restart_render_event.clear()
                    iteration = 0

                    if self.is_resized:
                        if not self.rpr_context.gl_interop:
                            # When gl_interop is not enabled, than resize is better to do in
                            # this thread. This is important for hybrid.
                            with self.render_lock:
                                self.rpr_context.resize(self.width, self.height)
                        self.is_resized = False

                    self.rpr_context.sync_auto_adapt_subdivision()
                    self.rpr_context.sync_portal_lights()
                    time_begin = time.perf_counter()
                    log(f"Restart render [{self.width}, {self.height}]")

                # rendering
                with self.render_lock:
                    if self.restart_render_event.is_set():
                        break

                    self.rpr_context.set_parameter(pyrpr.CONTEXT_FRAMECOUNT, iteration)
                    update_iterations = 1 if iteration == 0 else \
                        min(32, self.render_iterations - iteration)
                    self.rpr_context.set_parameter(pyrpr.CONTEXT_ITERATIONS, update_iterations)
                    self.rpr_context.render(restart=(iteration == 0))

                # resolving
                with self.resolve_lock:
                    self._resolve()

                self.is_rendered = True
                self.is_denoised = False
                iteration += update_iterations
                time_render = time.perf_counter() - time_begin

                if self.render_iterations > 0:
                    info_str = f"Time: {time_render:.1f} sec" \
                               f" | Iteration: {iteration}/{self.render_iterations}"
                else:
                    info_str = f"Time: {time_render:.1f}/{self.render_time} sec" \
                               f" | Iteration: {iteration}"

                if is_adaptive_active:
                    active_pixels = self.rpr_context.get_info(pyrpr.CONTEXT_ACTIVE_PIXEL_COUNT,
                                                              int)
                    adaptive_progress = max((all_pixels - active_pixels) / all_pixels, 0.0)
                    info_str += f" | Adaptive Sampling: {math.floor(adaptive_progress * 100)}%"

                if self.render_iterations > 0:
                    if iteration >= self.render_iterations:
                        is_last_iteration = True
                else:
                    if time_render >= self.render_time:
                        is_last_iteration = True
                if is_adaptive and active_pixels == 0:
                    is_last_iteration = True

                if is_last_iteration:
                    break

                self.is_intermediate_render = False
                self.notify_status(info_str, "Render")

            # notifying viewport that rendering is finished
            if is_last_iteration:
                time_render = time.perf_counter() - time_begin

                if self.image_filter:
                    self.notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}"
                                       f" | Denoising...", "Render")

                    # applying denoising
                    with self.resolve_lock:
                        if self.image_filter:
                            self.update_image_filter_inputs()
                            self.image_filter.run()
                            self.is_denoised = True

                    time_render = time.perf_counter() - time_begin
                    self.notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}"
                                       f" | Denoised", "Rendering Done")

                else:
                    self.notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}",
                                       "Rendering Done")

    def draw(self, context):
        log("Draw")

        if not self.is_synced or self.is_finished:
            return

        # initializing self.viewport_settings and requesting first self.restart_render_event
        if not self.viewport_settings:
            self.viewport_settings = ViewportSettings(context)

            self.viewport_settings.export_camera(self.rpr_context.scene.camera)
            self._resize(self.viewport_settings.width, self.viewport_settings.height)
            self.restart_render_event.set()

        if self.rendered_image is None:
            return

        im = self.rendered_image
        if self.gl_texture.width != im.shape[1] or self.gl_texture.height != im.shape[0]:
            self.gl_texture = gl.GLTexture(im.shape[1], im.shape[0])

        self.gl_texture.set_image(im)
        self.draw_texture(self.gl_texture.texture_id, context.scene)

        # checking for viewport updates: setting camera position and resizing
        viewport_settings = ViewportSettings(context)

        if viewport_settings.width * viewport_settings.height == 0:
            return

        if self.viewport_settings != viewport_settings:
            self.abort_render_iteration = True
            with self.render_lock:
                self.viewport_settings = viewport_settings
                self.viewport_settings.export_camera(self.rpr_context.scene.camera)
                self._resize(self.viewport_settings.width, self.viewport_settings.height)
                self.restart_render_event.set()

    def sync_update(self, context, depsgraph):
        """ sync just the updated things """

        if not self.is_synced:
            return

        if context.selected_objects != self.selected_objects:
            # only a selection change
            self.selected_objects = context.selected_objects
            return

        frame_current = depsgraph.scene.frame_current

        # get supported updates and sort by priorities
        updates = []
        for obj_type in (bpy.types.Scene, bpy.types.World, bpy.types.Material, bpy.types.Object, bpy.types.Collection):
            updates.extend(update for update in depsgraph.updates if isinstance(update.id, obj_type))

        sync_collection = False
        sync_world = False
        is_updated = False
        is_obj_updated = False

        material_override = depsgraph.view_layer.material_override

        shading_data = ShadingData(context)
        if self.shading_data != shading_data:
            sync_world = True

            if self.shading_data.use_scene_lights != shading_data.use_scene_lights:
                sync_collection = True

            self.shading_data = shading_data

        self.rpr_context.blender_data['depsgraph'] = depsgraph

        # if view mode changed need to sync collections
        mode_updated = False
        if self.view_mode != context.mode:
            self.view_mode = context.mode
            mode_updated = True

        if not updates:
            return

        self.abort_render_iteration = True
        with self.render_lock:
            for update in updates:
                obj = update.id
                log("sync_update", obj)
                if isinstance(obj, bpy.types.Scene):
                    is_updated |= self.update_render(obj, depsgraph.view_layer)

                    # Outliner object visibility change will provide us only bpy.types.Scene update
                    # That's why we need to sync objects collection in the end
                    sync_collection = True
                    continue

                if isinstance(obj, bpy.types.Material):
                    is_updated |= self.update_material_on_scene_objects(obj, depsgraph)
                    continue

                if isinstance(obj, bpy.types.Object):
                    if obj.type == 'CAMERA':
                        continue

                    indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)
                    active_and_mode_changed = mode_updated and context.active_object == obj.original
                    is_updated |= object.sync_update(self.rpr_context, obj,
                                                     update.is_updated_geometry or active_and_mode_changed,
                                                     update.is_updated_transform,
                                                     indirect_only=indirect_only,
                                                     material_override=material_override,
                                                     frame_current=frame_current)
                    is_obj_updated |= is_updated
                    continue

                if isinstance(obj, bpy.types.World):
                    sync_world = True

                if isinstance(obj, bpy.types.Collection):
                    sync_collection = True
                    continue

            if sync_world:
                world_settings = self._get_world_settings(depsgraph)
                if self.world_settings != world_settings:
                    self.world_settings = world_settings
                    self.world_settings.export(self.rpr_context)
                    is_updated = True

            if sync_collection:
                is_updated |= self.sync_objects_collection(depsgraph)

            if is_obj_updated:
                with self.resolve_lock:
                    self.rpr_context.sync_catchers()

        self.restart_render_event.set()