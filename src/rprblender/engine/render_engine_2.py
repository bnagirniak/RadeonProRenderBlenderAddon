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

from rprblender import RenderEngine
from rprblender.engine.context import RPRContext2

from rprblender.utils import logging
log = logging.Log(tag='RenderEngine2')


class RenderEngine2(RenderEngine):
    _RPRContext = RPRContext2

    def _update_athena_data(self, data):
        data['Quality'] = "rpr2"

    def render(self):
        resolve_event = threading.Event()
        is_finished = False

        def render_update_callback(progress):
            resolve_event.set()

        def do_resolve():
            log('Start do_resolve')
            while True:
                resolve_event.wait()

                if is_finished or self.rpr_engine.test_break():
                    break

                self.rpr_context.resolve()
                self.update_render_result((0, 0), (self.width, self.height),
                                          layer_name=self.render_layer_name)

            log('Finish do_resolve')

        self.rpr_context.set_render_update_callback(render_update_callback)
        resolve_thread = threading.Thread(target=do_resolve)
        resolve_thread.start()

        try:
            super().render()

        finally:
            is_finished = True
            resolve_event.set()
            resolve_thread.join()
