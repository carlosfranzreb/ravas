# -*- mode: python ; coding: utf-8 -*-


datas = [
    ('stream_processing/models/avatar/face_landmarker.task', 'stream_processing/models/avatar'),
    ('configs', 'configs'),
    ('onnx', 'onnx'),
    ('target_feats', 'target_feats'),
    ('../rpm/dist', 'rpm/dist'),
]

hidden_imports = [
    'stream_processing.models'
]


binaries = []

# FIX include mediapipe binaries/models/resources for face_detection & face_landmark:
import os
import inspect
import mediapipe
mediapipe_lib = os.path.dirname(inspect.getfile(mediapipe))
for sub_path in ["modules/face_landmark", "modules/face_detection"]:
    for f in os.listdir(os.path.join(mediapipe_lib, sub_path)):
        fp = os.path.join(mediapipe_lib, sub_path, f)
        if os.path.isdir(fp):
            continue
        binaries.append((fp, "mediapipe/" + sub_path))

# FIX avoid warning message for onnx runtime by including the DLLs
from PyInstaller.utils.hooks import collect_dynamic_libs
binaries.extend(collect_dynamic_libs('onnxruntime', destdir='onnxruntime/capi'))


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
