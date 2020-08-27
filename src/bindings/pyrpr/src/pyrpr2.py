# **********************************************************************
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
# ********************************************************************
import pyrpr

# will be used in other modules to check if RPR 2 is enabled
enabled = False


class Context(pyrpr.Context):
    def set_render_update_callback(self, func):
        pyrpr.ContextSetParameterByKeyPtr(self, pyrpr.CONTEXT_RENDER_UPDATE_CALLBACK_FUNC, func)
        pyrpr.ContextSetParameterByKeyPtr(self, pyrpr.CONTEXT_RENDER_UPDATE_CALLBACK_DATA,
                                          pyrpr.ffi.NULL)


class SphereLight(pyrpr.Light):
    def __init__(self, context):
        super().__init__(context)
        pyrpr.ContextCreateSphereLight(self.context, self)

        # keep target intensity and radius to adjust actual intensity when they are changed
        self._radius_squared = 1

    def set_radiant_power(self, r, g, b):
        # Adjust intensity by current radius
        pyrpr.SphereLightSetRadiantPower3f(self,
                                           r / self._radius_squared,
                                           g / self._radius_squared,
                                           b / self._radius_squared)

    def set_radius(self, radius):
        radius = max(radius, 0.01)
        self._radius_squared = radius * radius
        pyrpr.SphereLightSetRadius(self, radius)


class DiskLight(pyrpr.Light):
    def __init__(self, context):
        super().__init__(context)
        pyrpr.ContextCreateDiskLight(self.context, self)

        # keep target intensity and radius to adjust actual intensity when they are changed
        self._radius_squared = 1

    def set_radiant_power(self, r, g, b):
        # Adjust intensity by current radius
        pyrpr.DiskLightSetRadiantPower3f(self,
                                   r / self._radius_squared,
                                   g / self._radius_squared,
                                   b / self._radius_squared)

    def set_cone_shape(self, iangle, oangle):
        # Use external angle oangle
        pyrpr.DiskLightSetAngle(self, oangle)

    def set_radius(self, radius):
        radius = max(radius, 0.01)
        self._radius_squared = radius * radius
        pyrpr.DiskLightSetRadius(self, radius)
