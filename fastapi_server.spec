# -*- mode: python ; coding: utf-8 -*-
# fastapi_server.spec — PyInstaller --onedir，包含所有 ML 套件

from PyInstaller.utils.hooks import collect_all, collect_submodules, collect_data_files

block_cipher = None

# 收集動態載入的套件（避免遺漏 C++ 擴充、yaml 設定等）
_datas, _binaries, _hidden = [], [], []

for pkg in ("paddlepaddle", "paddleocr", "paddle", "paddlex", "easyocr", "cv2"):
    try:
        d, b, h = collect_all(pkg)
        _datas += d; _binaries += b; _hidden += h
    except Exception:
        pass

# paddlex configs (YAML pipeline/module 設定) 需要額外收集
_datas += collect_data_files("paddlex", includes=["**/*.yaml", "**/*.yml", "**/*.json"])

_hidden += [
    # uvicorn
    "uvicorn.logging", "uvicorn.loops", "uvicorn.loops.auto",
    "uvicorn.protocols", "uvicorn.protocols.http", "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets", "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan", "uvicorn.lifespan.on",
    # fastapi / starlette
    "fastapi", "starlette", "anyio", "anyio._backends._asyncio",
    # ML
    "torch", "torchvision",
    "cv2", "easyocr", "PIL", "numpy",
]

# transformers 用 importlib 動態載入所有子模組，需全部打包
_hidden += collect_submodules("transformers")

a = Analysis(
    ["fastapi_server.py"],
    pathex=["."],
    binaries=_binaries,
    datas=_datas,
    hiddenimports=_hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fastapi_server",
    debug=False,
    strip=False,
    upx=True,
    console=True,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="OCR_api",
)
