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
from .viewport_engine_base import (
    ViewportEngineBase, ViewLayerSettings, ViewportSettings, ShadingData, draw_texture,
    MIN_ADAPT_RATIO_DIFF, MIN_ADAPT_RESOLUTION_RATIO_DIFF
)
from rprblender.export import camera, material, world, object, instance, particle
from rprblender.export.mesh import assign_materials
from rprblender.utils import gl
from rprblender import utils
from rprblender.utils.user_settings import get_user_settings

from rprblender.utils import logging
log = logging.Log(tag='viewport_engine')


class ViewportEngine(ViewportEngineBase):
    """ Viewport render engine """

    TYPE = 'VIEWPORT'

    def __init__(self, rpr_engine):
        super().__init__(rpr_engine)

        self.viewport_settings: ViewportSettings = None
        self.world_settings: world.WorldData = None
        self.shading_data: ShadingData = None
        self.view_layer_data: ViewLayerSettings = None
        self.view_mode = None
        self.selected_objects = None

        self.sync_render_thread: threading.Thread = None
        self.restart_render_event = threading.Event()
        self.render_lock = threading.Lock()
        self.resolve_lock = threading.Lock()

        self.is_finished = False
        self.is_synced = False
        self.is_rendered = False
        self.is_denoised = False
        self.is_resized = False

        self.requested_adapt_ratio = None
        self.is_resolution_adapted = False
        self.width = 1
        self.height = 1

        self.render_iterations = 0
        self.render_time = 0

    def stop_render(self):
        self.is_finished = True
        self.restart_render_event.set()
        self.sync_render_thread.join()

        self.rpr_context = None
        self.image_filter = None

    def _resolve(self):
        self.rpr_context.resolve()

    def _do_sync_render(self, depsgraph):
        """
        Thread function for self.sync_render_thread. It always run during viewport render.
        If it doesn't render it waits for self.restart_render_event
        """

        def notify_status(info, status):
            """ Display export progress status """
            wrap_info = textwrap.fill(info, 120)
            self.rpr_engine.update_stats(status, wrap_info)
            log(status, wrap_info)

            # requesting blender to call draw()
            self.rpr_engine.tag_redraw()

        class FinishRender(Exception):
            pass

        try:
            # SYNCING OBJECTS AND INSTANCES
            notify_status("Starting...", "Sync")
            time_begin = time.perf_counter()

            # exporting objects
            frame_current = depsgraph.scene.frame_current
            material_override = depsgraph.view_layer.material_override
            objects_len = len(depsgraph.objects)
            for i, obj in enumerate(self.depsgraph_objects(depsgraph)):
                if self.is_finished:
                    raise FinishRender

                time_sync = time.perf_counter() - time_begin
                notify_status(f"Time {time_sync:.1f} | Object ({i}/{objects_len}): {obj.name}", "Sync")

                indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)
                object.sync(self.rpr_context, obj,
                            indirect_only=indirect_only, material_override=material_override,
                            frame_current=frame_current)

            # exporting instances
            instances_len = len(depsgraph.object_instances)
            last_instances_percent = 0

            for i, inst in enumerate(self.depsgraph_instances(depsgraph)):
                if self.is_finished:
                    raise FinishRender

                instances_percent = (i * 100) // instances_len
                if instances_percent > last_instances_percent:
                    time_sync = time.perf_counter() - time_begin
                    notify_status(f"Time {time_sync:.1f} | Instances {instances_percent}%", "Sync")
                    last_instances_percent = instances_percent

                indirect_only = inst.parent.original.indirect_only_get(view_layer=depsgraph.view_layer)
                instance.sync(self.rpr_context, inst,
                              indirect_only=indirect_only, material_override=material_override,
                              frame_current=frame_current)

            # shadow catcher
            self.rpr_context.sync_catchers(depsgraph.scene.render.film_transparent)

            self.is_synced = True

            # RENDERING
            notify_status("Starting...", "Render")

            is_adaptive = self.rpr_context.is_aov_enabled(pyrpr.AOV_VARIANCE)

            # Infinite cycle, which starts when scene has to be re-rendered.
            # It waits for restart_render_event be enabled.
            # Exit from this cycle is implemented through raising FinishRender
            # when self.is_finished be enabled from main thread.
            while True:
                self.restart_render_event.wait()

                if self.is_finished:
                    raise FinishRender

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
                        raise FinishRender

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
                        self.rpr_context.render(restart=(iteration == 0))

                    # resolving
                    with self.resolve_lock:
                        self._resolve()

                    self.is_rendered = True
                    self.is_denoised = False
                    iteration += 1

                    # checking for last iteration
                    # preparing information to show in viewport
                    time_render_prev = time_render
                    time_render = time.perf_counter() - time_begin
                    iteration_time = time_render - time_render_prev
                    if self.user_settings.adapt_viewport_resolution \
                            and not self.is_resolution_adapted \
                            and iteration == 2:
                        target_time = 1.0 / self.user_settings.viewport_samples_per_sec
                        self.requested_adapt_ratio = target_time / iteration_time

                    if self.render_iterations > 0:
                        info_str = f"Time: {time_render:.1f} sec"\
                                   f" | Iteration: {iteration}/{self.render_iterations}"
                    else:
                        info_str = f"Time: {time_render:.1f}/{self.render_time} sec"\
                                   f" | Iteration: {iteration}"

                    if is_adaptive_active:
                        active_pixels = self.rpr_context.get_info(pyrpr.CONTEXT_ACTIVE_PIXEL_COUNT, int)
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

                    notify_status(info_str, "Render")

                # notifying viewport that rendering is finished
                if is_last_iteration:
                    time_render = time.perf_counter() - time_begin

                    if self.image_filter:
                        notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}"
                                      f" | Denoising...", "Render")

                        # applying denoising
                        with self.resolve_lock:
                            if self.image_filter:
                                self.update_image_filter_inputs()
                                self.image_filter.run()
                                self.is_denoised = True

                        time_render = time.perf_counter() - time_begin
                        notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}"
                                      f" | Denoised", "Rendering Done")

                    else:
                        notify_status(f"Time: {time_render:.1f} sec | Iteration: {iteration}",
                                      "Rendering Done")

        except FinishRender:
            log("Finish by user")

        except Exception as e:
            log.error(e, 'EXCEPTION:', traceback.format_exc())
            self.is_finished = True

            # notifying viewport about error
            notify_status(f"{e}.\nPlease see logs for more details.", "ERROR")

        log("Finish _do_sync_render")

    def sync(self, context, depsgraph):
        log('Start sync')

        scene = depsgraph.scene
        viewport_limits = scene.rpr.viewport_limits
        view_layer = depsgraph.view_layer
        settings = get_user_settings()
        use_gl_interop = settings.use_gl_interop and not scene.render.film_transparent

        scene.rpr.init_rpr_context(self.rpr_context, is_final_engine=False,
                                   use_gl_interop=use_gl_interop)

        self.rpr_context.blender_data['depsgraph'] = depsgraph

        self.shading_data = ShadingData(context)
        self.view_layer_data = ViewLayerSettings(view_layer)

        # setting initial render resolution as (1, 1) just for AOVs creation.
        # It'll be resized to correct resolution in draw() function
        self.rpr_context.resize(1, 1)
        if not self.rpr_context.gl_interop:
            self.gl_texture = gl.GLTexture(self.width, self.height)

        self.rpr_context.enable_aov(pyrpr.AOV_COLOR)

        if viewport_limits.noise_threshold > 0.0:
            # if adaptive is enable turn on aov and settings
            self.rpr_context.enable_aov(pyrpr.AOV_VARIANCE)
            viewport_limits.set_adaptive_params(self.rpr_context)

        self.rpr_context.scene.set_name(scene.name)

        self.world_settings = self._get_world_settings(depsgraph)
        self.world_settings.export(self.rpr_context)

        rpr_camera = self.rpr_context.create_camera()
        rpr_camera.set_name("Camera")
        self.rpr_context.scene.set_camera(rpr_camera)

        # image filter
        image_filter_settings = view_layer.rpr.denoiser.get_settings(scene, False)
        image_filter_settings['resolution'] = (self.width, self.height)
        self.setup_image_filter(image_filter_settings)

        # other context settings
        self.rpr_context.set_parameter(pyrpr.CONTEXT_PREVIEW, True)
        self.rpr_context.set_parameter(pyrpr.CONTEXT_ITERATIONS, 1)
        scene.rpr.export_render_mode(self.rpr_context)
        scene.rpr.export_ray_depth(self.rpr_context)
        scene.rpr.export_pixel_filter(self.rpr_context)

        self.render_iterations, self.render_time = (viewport_limits.max_samples, 0)

        self.is_finished = False
        self.restart_render_event.clear()

        self.view_mode = context.mode
        self.space_data = context.space_data
        self.selected_objects = context.selected_objects
        self.sync_render_thread = threading.Thread(target=self._do_sync_render, args=(depsgraph,))
        self.sync_render_thread.start()

        log('Finish sync')

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

        with self.render_lock:
            for update in updates:
                obj = update.id
                log("sync_update", obj)
                if isinstance(obj, bpy.types.Scene):
                    is_updated |= self.update_render(obj, depsgraph.view_layer)

                    # Outliner object visibility change will provide us only bpy.types.Scene update
                    # That's why we need to sync objects collection in the end
                    sync_collection = True

                    if is_updated:
                        self.is_resolution_adapted = False

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

        if is_updated:
            self.restart_render_event.set()

    def _get_render_image(self):
        ''' This is only called for non-GL interop image gets '''
        if utils.IS_MAC:
            with self.render_lock:
                return self.rpr_context.get_image()
        else:
            return self.rpr_context.get_image()

    def draw(self, context):
        log("Draw")

        if not self.is_synced or self.is_finished:
            return

        scene = context.scene

        # initializing self.viewport_settings and requesting first self.restart_render_event
        with self.render_lock:
            if not self.viewport_settings:
                self.viewport_settings = ViewportSettings(context)

                self.viewport_settings.export_camera(self.rpr_context.scene.camera)
                self._resize(self.viewport_settings.width, self.viewport_settings.height)
                self.is_resolution_adapted = False
                self.restart_render_event.set()

        if not self.is_rendered:
            return

        # drawing functionality
        def draw_(texture_id):
            if scene.rpr.render_mode in ('WIREFRAME', 'MATERIAL_INDEX',
                                         'POSITION', 'NORMAL', 'TEXCOORD'):
                # Draw without color management
                draw_texture_2d(texture_id, self.viewport_settings.border[0],
                                *self.viewport_settings.border[1])

            else:
                # Bind shader that converts from scene linear to display space,
                bgl.glEnable(bgl.GL_BLEND)
                bgl.glBlendFunc(bgl.GL_ONE, bgl.GL_ONE_MINUS_SRC_ALPHA)
                self.rpr_engine.bind_display_space_shader(scene)

                # note this has to draw to region size, not scaled down size
                draw_texture(texture_id, *self.viewport_settings.border[0],
                             *self.viewport_settings.border[1])

                self.rpr_engine.unbind_display_space_shader()
                bgl.glDisable(bgl.GL_BLEND)

        def draw__():
            if self.is_denoised:
                im = None
                with self.resolve_lock:
                    if self.image_filter:
                        im = self.image_filter.get_data()

                if im is not None:
                    self.gl_texture.set_image(im)
                    draw_(self.gl_texture.texture_id)
                    return

            if self.rpr_context.gl_interop:
                with self.resolve_lock:
                    draw_(self.rpr_context.get_frame_buffer().texture_id)
                return

            with self.resolve_lock:
                im = self._get_render_image()

            self.gl_texture.set_image(im)
            draw_(self.gl_texture.texture_id)

        draw__()

        # checking for viewport updates: setting camera position and resizing
        with self.render_lock:
            viewport_settings = ViewportSettings(context)

            if viewport_settings.width * viewport_settings.height == 0:
                return

            if self.viewport_settings != viewport_settings:
                self.viewport_settings = viewport_settings
                self.viewport_settings.export_camera(self.rpr_context.scene.camera)
                if self.user_settings.adapt_viewport_resolution:
                    # trying to use previous resolution or almost same pixels number
                    max_w, max_h = self.viewport_settings.width, self.viewport_settings.height
                    min_w = max(max_w * self.user_settings.min_viewport_resolution_scale // 100, 1)
                    min_h = max(max_h * self.user_settings.min_viewport_resolution_scale // 100, 1)
                    w, h = self.rpr_context.width, self.rpr_context.height

                    if abs(w / h - max_w / max_h) > MIN_ADAPT_RESOLUTION_RATIO_DIFF:
                        scale = math.sqrt(w * h / (max_w * max_h))
                        w, h = int(max_w * scale), int(max_h * scale)

                    self._resize(min(max(w, min_w), max_w),
                                 min(max(h, min_h), max_h))
                else:
                    self._resize(self.viewport_settings.width, self.viewport_settings.height)

                self.is_resolution_adapted = False
                self.restart_render_event.set()

            else:
                if self.requested_adapt_ratio is not None:
                    max_w, max_h = self.viewport_settings.width, self.viewport_settings.height
                    min_w = max(max_w * self.user_settings.min_viewport_resolution_scale // 100, 1)
                    min_h = max(max_h * self.user_settings.min_viewport_resolution_scale // 100, 1)
                    if abs(1.0 - self.requested_adapt_ratio) > MIN_ADAPT_RATIO_DIFF:
                        scale = math.sqrt(self.requested_adapt_ratio)
                        w, h = int(self.rpr_context.width * scale),\
                               int(self.rpr_context.height * scale)
                    else:
                        w, h = self.rpr_context.width, self.rpr_context.height

                    self._resize(min(max(w, min_w), max_w),
                                 min(max(h, min_h), max_h))

                    self.requested_adapt_ratio = None
                    self.is_resolution_adapted = True

                elif not self.user_settings.adapt_viewport_resolution:
                    self._resize(self.viewport_settings.width, self.viewport_settings.height)

                if self.is_resized:
                    self.restart_render_event.set()

    def _resize(self, width, height):
        if self.width == width and self.height == height:
            self.is_resized = False
            return

        self.width = width
        self.height = height

        if self.rpr_context.gl_interop:
            # GL framebuffer ahs to be recreated in this thread,
            # that's why we call resize here
            with self.resolve_lock:
                self.rpr_context.resize(self.width, self.height)

        if self.gl_texture:
            self.gl_texture = gl.GLTexture(self.width, self.height)

        if self.image_filter:
            image_filter_settings = self.image_filter.settings.copy()
            image_filter_settings['resolution'] = self.width, self.height
            self.setup_image_filter(image_filter_settings)

        if self.world_settings.backplate:
            self.world_settings.backplate.export(self.rpr_context, (self.width, self.height))

        self.is_resized = True

    def sync_objects_collection(self, depsgraph):
        """
        Removes objects which are not present in depsgraph anymore.
        Adds objects which are not present in rpr_context but existed in depsgraph
        """
        res = False
        view_layer_data = ViewLayerSettings(depsgraph.view_layer)
        material_override = view_layer_data.material_override

        # set of depsgraph object keys
        depsgraph_keys = set.union(
            set(object.key(obj) for obj in self.depsgraph_objects(depsgraph)),
            set(instance.key(obj) for obj in self.depsgraph_instances(depsgraph))
        )

        # set of visible rpr object keys
        rpr_object_keys = set(key for key, obj in self.rpr_context.objects.items()
                              if not isinstance(obj, pyrpr.Shape) or obj.is_visible)

        # sets of objects keys to remove from rpr
        object_keys_to_remove = rpr_object_keys - depsgraph_keys

        # sets of objects keys to export into rpr
        object_keys_to_export = depsgraph_keys - rpr_object_keys

        if object_keys_to_remove:
            log("Object keys to remove", object_keys_to_remove)
            for obj_key in object_keys_to_remove:
                if obj_key in self.rpr_context.objects:
                    self.rpr_context.remove_object(obj_key)
                    res = True

        if object_keys_to_export:
            log("Object keys to add", object_keys_to_export)

            res |= self.sync_collection_objects(depsgraph, object_keys_to_export,
                                                material_override)

            res |= self.sync_collection_instances(depsgraph, object_keys_to_export,
                                                  material_override)

        # update/remove material override on rest of scene object
        if view_layer_data != self.view_layer_data:
            # update/remove material override on all other objects
            self.view_layer_data = view_layer_data
            res = True

            rpr_mesh_keys = set(key for key, obj in self.rpr_context.objects.items()
                                if isinstance(obj, pyrpr.Mesh) and obj.is_visible)
            unchanged_meshes_keys = tuple(e for e in depsgraph_keys if e in rpr_mesh_keys)
            log("Object keys to update material override", unchanged_meshes_keys)
            self.sync_collection_objects(depsgraph, unchanged_meshes_keys,
                                         material_override)

            self.sync_collection_instances(depsgraph, unchanged_meshes_keys,
                                           material_override)

        return res

    def sync_collection_objects(self, depsgraph, object_keys_to_export, material_override):
        """ Export collections objects """
        res = False
        frame_current = depsgraph.scene.frame_current

        for obj in self.depsgraph_objects(depsgraph):
            obj_key = object.key(obj)
            if obj_key not in object_keys_to_export:
                continue

            rpr_obj = self.rpr_context.objects.get(obj_key, None)
            if rpr_obj:
                rpr_obj.set_visibility(True)

                if not material_override:
                    rpr_obj.set_material(None)
                assign_materials(self.rpr_context, rpr_obj, obj, material_override)
                res = True
            else:
                indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)
                object.sync(self.rpr_context, obj,
                            indirect_only=indirect_only, material_override=material_override,
                            frame_current=frame_current)

                res = True
        return res

    def sync_collection_instances(self, depsgraph, object_keys_to_export, material_override):
        """ Export collections instances """
        res = False
        frame_current = depsgraph.scene.frame_current

        for inst in self.depsgraph_instances(depsgraph):
            instance_key = instance.key(inst)
            if instance_key not in object_keys_to_export:
                continue

            if not material_override:
                inst_obj = self.rpr_context.objects.get(instance_key, None)
                if inst_obj:
                    if len(inst.object.material_slots) == 0:
                        # remove override from instance without assigned materials
                        inst_obj.set_material(None)
                    assign_materials(self.rpr_context, inst_obj, inst.object)
                    res = True
            else:
                indirect_only = inst.parent.original.indirect_only_get(view_layer=depsgraph.view_layer)
                instance.sync(self.rpr_context, inst,
                              indirect_only=indirect_only, material_override=material_override,
                              frame_current=frame_current)
                res = True
        return res

    def update_material_on_scene_objects(self, mat, depsgraph):
        """ Find all mesh material users and reapply material """
        material_override = depsgraph.view_layer.material_override
        frame_current = depsgraph.scene.frame_current

        if material_override and material_override.name == mat.name:
            objects = self.depsgraph_objects(depsgraph)
            active_mat = material_override
        else:
            objects = tuple(obj for obj in self.depsgraph_objects(depsgraph)
                            if mat.name in obj.material_slots.keys())
            active_mat = mat

        updated = False
        for obj in objects:
            rpr_material = material.sync_update(self.rpr_context, active_mat, obj=obj)
            rpr_volume = material.sync_update(self.rpr_context, active_mat, 'Volume', obj=obj)
            rpr_displacement = material.sync_update(self.rpr_context, active_mat, 'Displacement', obj=obj)

            if not rpr_material and not rpr_volume and not rpr_displacement:
                continue

            indirect_only = obj.original.indirect_only_get(view_layer=depsgraph.view_layer)

            if object.key(obj) not in self.rpr_context.objects:
                object.sync(self.rpr_context, obj, indirect_only=indirect_only,
                            frame_current=frame_current)
                updated = True
                continue

            updated |= object.sync_update(self.rpr_context, obj, False, False,
                                          indirect_only=indirect_only,
                                          material_override=material_override,
                                          frame_current=frame_current)

        return updated

    def update_render(self, scene: bpy.types.Scene, view_layer: bpy.types.ViewLayer):
        ''' update settings if changed while live returns True if restart needed '''
        restart = scene.rpr.export_render_mode(self.rpr_context)
        restart |= scene.rpr.export_ray_depth(self.rpr_context)
        restart |= scene.rpr.export_pixel_filter(self.rpr_context)

        render_iterations, render_time = (scene.rpr.viewport_limits.max_samples, 0)

        if self.render_iterations != render_iterations or self.render_time != render_time:
            self.render_iterations = render_iterations
            self.render_time = render_time
            restart = True

        restart |= scene.rpr.viewport_limits.set_adaptive_params(self.rpr_context)

        # image filter
        image_filter_settings = view_layer.rpr.denoiser.get_settings(scene, False)
        image_filter_settings['resolution'] = (self.rpr_context.width, self.rpr_context.height)
        if self.setup_image_filter(image_filter_settings):
            self.is_denoised = False
            restart = True

        return restart

    def setup_image_filter(self, settings):
        with self.resolve_lock:
            return super().setup_image_filter(settings)

    def _enable_image_filter(self, settings):
        super()._enable_image_filter(settings)

        if not self.gl_texture:
            self.gl_texture = gl.GLTexture(self.rpr_context.width, self.rpr_context.height)
