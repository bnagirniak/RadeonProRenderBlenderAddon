import os

import numpy as np
import math

import bpy

from rprblender.engine.context import RPRContext
from rprblender.properties.light import MAX_LUMINOUS_EFFICACY
from . import mesh, image, object
from rprblender.utils.conversion import convert_kelvins_to_rgb
from rprblender import utils

from rprblender.utils import logging
log = logging.Log(tag='export.light')


def get_radiant_power(light: bpy.types.Light, area=0.0):
    """ Return light radiant power depending of light type and selected units """

    rpr = light.rpr

    # calculating color intensity
    color = np.array(light.color)
    if rpr.use_temperature:
        color *= convert_kelvins_to_rgb(rpr.temperature)
    intensity = color * rpr.intensity
    default_intensity = color * light.energy

    # calculating radian power for core
    if light.type in ('POINT', 'SPOT'):
        units = rpr.intensity_units_point
        if units == 'DEFAULT':
            return default_intensity / (4*math.pi)  # dividing by 4*pi to be more convenient with cycles point light

        # converting to lumen
        if units == 'LUMEN':
            lumen = intensity
        elif units == 'WATTS':
            lumen = intensity * rpr.luminous_efficacy
        else:
            raise ValueError("Incorrect light units value", light, units)

        return lumen / MAX_LUMINOUS_EFFICACY

    elif light.type == 'SUN':
        units = rpr.intensity_units_dir
        if units == 'DEFAULT':
            return default_intensity * 0.01         # multiplying by 0.01 to be more convenient with point light

        # converting to luminance
        if units == 'LUMINANCE':
            luminance = intensity
        elif units == 'RADIANCE':
            luminance = intensity * rpr.luminous_efficacy
        else:
            raise ValueError("Incorrect light units value", light, units)

        return luminance / MAX_LUMINOUS_EFFICACY

    elif light.type == 'AREA':
        units = rpr.intensity_units_area
        if units == 'DEFAULT':
            if rpr.intensity_normalization:
                return default_intensity / area
            return default_intensity

        # converting to luminance
        if units == 'LUMEN':
            luminance = intensity / area
        elif units == 'WATTS':
            luminance = intensity * rpr.luminous_efficacy / area
        elif units == 'LUMINANCE':
            luminance = intensity
        elif units == 'RADIANCE':
            luminance = intensity * rpr.luminous_efficacy
        else:
            raise ValueError("Incorrect light units value", light, units)

        return luminance / MAX_LUMINOUS_EFFICACY


def sync_ies_light(rpr_context, light: bpy.types.Light, light_key) -> RPRContext._IESLight:
    """ Sync IES light source """
    if light.rpr.ies_file.source not in ('FILE', 'GENERATED'):
        # unsupported image source type
        return rpr_context.create_light(light_key, 'point')

    file_path = image.cache_image_file(light.rpr.ies_file)
    if not file_path:
        rpr_context.create_empty_object(light_key)
        return None

    rpr_light = rpr_context.create_light(light_key, 'ies')
    rpr_light.set_image_from_file(file_path, 256, 256)

    return rpr_light


def sync(rpr_context: RPRContext, obj: bpy.types.Object, instance_key=None):
    """ Creates pyrpr.Light from obj.data: bpy.types.Light """

    from rprblender.engine.preview_engine import PreviewEngine

    light = obj.data
    rpr = light.rpr
    log("sync", light, obj)

    area = 0.0
    light_key = object.key(obj) if not instance_key else instance_key

    if light.type == 'POINT':
        if light.rpr.ies_file:
            rpr_light = sync_ies_light(rpr_context, light, light_key)
        else:
            rpr_light = rpr_context.create_light(light_key, 'point')

    elif light.type in ('SUN', 'HEMI'):  # just in case old scenes will have outdated Hemi
        rpr_light = rpr_context.create_light(light_key, 'directional')
        rpr_light.set_shadow_softness_angle(light.rpr.shadow_softness_angle)

    elif light.type == 'SPOT':
        rpr_light = rpr_context.create_light(light_key, 'spot')
        oangle = 0.5 * light.spot_size  # half of spot_size
        iangle = oangle * (1.0 - light.spot_blend * light.spot_blend)  # square dependency of spot_blend
        rpr_light.set_cone_shape(iangle, oangle)

    elif light.type == 'AREA':
        data = mesh.MeshData.init_from_shape_type(rpr.shape, light.size, light.size_y, segments=32)
        area = data.area

        rpr_light = rpr_context.create_area_light(
            light_key,
            data.vertices, data.normals, data.uvs,
            data.vertex_indices, data.normal_indices, data.uv_indices,
            data.num_face_vertices
        )

        rpr_light.set_visibility(rpr.visible)
        rpr_light.set_shadow(rpr.visible and rpr.cast_shadows)

        if rpr.color_map:
            rpr_light.set_image(image.sync(rpr_context, rpr.color_map))

    else:
        raise ValueError("Unsupported light type", light, light.type)

    rpr_light.set_name(light.name)

    power = get_radiant_power(light, area)

    # Material Previews are overly bright, that's why
    # decreasing light intensity for material preview by 10 times
    if rpr_context.engine_type == PreviewEngine.TYPE:
        power /= 10.0

    rpr_light.set_radiant_power(*power)
    rpr_light.set_transform(object.get_transform(obj))
    rpr_light.set_group_id(1 if light.rpr.group == 'KEY' else 2)

    rpr_context.scene.attach(rpr_light)


def sync_update(rpr_context: RPRContext, obj: bpy.types.Object, is_updated_geometry, is_updated_transform) -> bool:
    """ Update existing light from obj.data: bpy.types.Light or create a new light """

    light = obj.data
    log("sync_update", light, obj, is_updated_geometry, is_updated_transform)

    light_key = object.key(obj)
    rpr_light = rpr_context.objects.get(light_key, None)

    if not rpr_light:
        # no such light => creating light
        sync(rpr_context, obj)
        return True

    if is_updated_geometry:
        # light exists, but its settings were changed => recreating light
        rpr_context.remove_object(light_key)
        sync(rpr_context, obj)
        # TODO: Better to set only changed parameters without recreating.
        #  But this idea has to be applied to other objects also with refactoring.
        return True

    if is_updated_transform:
        # updating only light transform
        rpr_light.set_transform(object.get_transform(obj))
        return True

    return False
