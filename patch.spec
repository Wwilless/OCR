# -*- mode: python ; coding: utf-8 -*-
# patch.spec — PyInstaller --onefile，輕量啟動器，不含 ML 套件

block_cipher = None

a = Analysis(
    ["patch.py"],
    pathex=["."],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # 明確排除所有 ML 套件，保持輕量
    excludes=[
        "torch", "torchvision", "easyocr", "transformers",
        "paddleocr", "paddlepaddle", "paddle",
        "cv2", "numpy", "PIL",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="patch",
    debug=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    icon=None,
)
