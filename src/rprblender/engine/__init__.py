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
import os
import sys

from rprblender import config
from rprblender import utils

from rprblender.utils import logging
log = logging.Log(tag='engine.init')


if utils.IS_DEBUG_MODE:
    project_root = utils.package_root_dir().parent.parent
    os.environ['PATH'] = f"{project_root / '.sdk/rpr/bin'};{project_root / '.sdk/rif/bin'};" \
                         f"{os.environ['PATH']}"
    sys.path.append(str(project_root / "src/bindings/pyrpr/.build"))
    sys.path.append(str(project_root / "src/bindings/pyrpr/src"))

else:
    os.environ['PATH'] = f"{utils.package_root_dir()};{os.environ['PATH']}"

import pyrpr
import pyhybrid
import pyrpr2

rpr_version = utils.core_ver_str(full=True)

log.info(f"Core version: {rpr_version}")
pyrpr.lib_wrapped_log_calls = config.pyrpr_log_calls
pyrpr.init(logging.Log(tag='core'))

import pyrpr_load_store
pyrpr_load_store.init()

import pyrprimagefilters
rif_version = utils.rif_ver_str(full=True)
log.info(f"RIF version: {rif_version}")
pyrprimagefilters.lib_wrapped_log_calls = config.pyrprimagefilters_log_calls
pyrprimagefilters.init(logging.Log(tag='rif'))

from rprblender.utils import helper_lib
helper_lib.init()


def register_plugins():
    rprsdk_bin_path = utils.package_root_dir() if not utils.IS_DEBUG_MODE else \
        utils.package_root_dir().parent.parent / '.sdk/rpr/bin'

    def register_plugin(ContextCls, lib_name, cache_path):
        lib_path = rprsdk_bin_path / lib_name
        ContextCls.register_plugin(lib_path, cache_path)
        log(f"Registered plugin: plugin_id={ContextCls.plugin_id}, "
                  f"lib_path={lib_path}, cache_path={cache_path}")

    cache_dir = utils.core_cache_dir()

    register_plugin(pyrpr.Context,
                    {'Windows': 'Tahoe64.dll',
                     'Linux': 'libTahoe64.so',
                     'Darwin': 'libTahoe64.dylib'}[utils.OS],
                    cache_dir / f"{hex(pyrpr.API_VERSION)}_rpr")

    # enabling hybrid only for Windows and Linux
    pyhybrid.enabled = config.enable_hybrid and (utils.IS_WIN or utils.IS_LINUX)
    if pyhybrid.enabled:
        try:
            register_plugin(pyhybrid.Context,
                            {'Windows': 'Hybrid.dll',
                             'Linux': 'Hybrid.so'}[utils.OS],
                            cache_dir / f"{hex(pyrpr.API_VERSION)}_hybrid")
        except RuntimeError as err:
            log.warn(err)
            pyhybrid.enabled = False

    # enabling RPR 2
    try:
        register_plugin(pyrpr2.Context,
                        {'Windows': 'Northstar64.dll',
                            'Linux': 'libNorthstar64.so',
                            'Darwin': 'libNorthstar64.dylib'}[utils.OS],
                        cache_dir / f"{hex(pyrpr.API_VERSION)}_rpr2")
    except RuntimeError as err:
        log.warn(err)


register_plugins()

pyrpr.Context.load_devices()
log(f"Loaded devices: cpu={pyrpr.Context.cpu_device}, gpu={pyrpr.Context.gpu_devices}")
