# -*- mode: python ; coding: utf-8 -*-


datas = [
    ('stream_processing/models/avatar/face_landmarker.task', 'stream_processing/models/avatar'),
    ('stream_processing/models/avatar/opengl/texture_light_skeleton_morph.glsl', 'stream_processing/models/avatar/opengl'),
    ('configs', 'configs'),
    ('onnx', 'onnx'),
    ('target_feats', 'target_feats'),
    ('../rpm/dist', 'rpm/dist'),
]

hidden_imports = [
    'stream_processing.models'
]


binaries = []


import inspect
import os
# HELPER for adding files from libraries to binaries-list:
def add_binaries_for(lib_name, lib_path, sub_paths, is_recursive):
    if not isinstance(lib_path, str):
        lib_path = os.path.dirname(inspect.getfile(lib_path))
    # print('add_binaries_for: ', lib_name, lib_path, sub_paths, is_recursive)
    for sub_path in sub_paths:
        for f in os.listdir(os.path.normpath(os.path.join(lib_path, sub_path))):
            fp = os.path.normpath(os.path.join(lib_path, sub_path, f))
            if os.path.isdir(fp):
                if is_recursive and f != '__pycache__':
                    add_binaries_for(lib_name, lib_path, [os.path.normpath(sub_path+'/'+f)], is_recursive)
                else:
                    continue
            binaries.append((fp, os.path.normpath(lib_name+'/' + sub_path)))


# FIX include mediapipe binaries/models/resources for face_detection & face_landmark:
import mediapipe
add_binaries_for('mediapipe', mediapipe, ["modules/face_landmark", "modules/face_detection"], False)

# FIX moderngl_window seems to be completely unsupported: add EVERYTHING (recursively) from moderngl_window directory:
import moderngl_window
add_binaries_for('moderngl_window', moderngl_window, ["./"], True)

# FIX avoid warning message for onnx runtime by including the DLLs
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries.extend(collect_dynamic_libs('onnxruntime', destdir='onnxruntime/capi'))

# print('binaries:\n' + '\n'.join([str(b) for b in binaries]))

a = Analysis(
    ['run_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='stream_processing_gui',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='stream_processing',
)
