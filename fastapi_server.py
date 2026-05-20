"""FastAPI 伺服器 — 直接 import capture_ocr，模型啟動時預載一次"""

import multiprocessing
import os
import socket
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# frozen exe (PyInstaller) 需要呼叫 freeze_support 才能正確產生子進程
if getattr(sys, "frozen", False):
    multiprocessing.freeze_support()

# 確保 capture_ocr.py 與 config_manager.py 在 import 路徑中
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import capture_ocr  # noqa: E402
from config_manager import load_config  # noqa: E402

API_KEY = "OCR-2026-KEY-a3f7d2b9e1c458f0"

_models_ready = False
_models_error: str = ""


# ── 工具函式 ────────────────────────────────────────────────────────────

def _find_port(start: int) -> int:
    for port in range(start, start + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Port {start}~{start+19} 均被占用，無法啟動")


def _check_models_present() -> tuple[bool, list[str]]:
    home = Path.home()
    missing: list[str] = []
    if not (home / ".EasyOCR" / "model").exists():
        missing.append("EasyOCR")
    hub = home / ".cache" / "huggingface" / "hub"
    if not hub.exists() or not any(hub.glob("models--microsoft--trocr-small-printed*")):
        missing.append("TrOCR")
    if not (home / ".paddleocr").exists():
        missing.append("PaddleOCR")
    return len(missing) == 0, missing


def _preload_models_and_exit() -> None:
    """--preload-models 模式：下載/載入所有模型後退出，供 patch.exe 呼叫"""
    print("模型載入/下載中（首次執行需要數分鐘）...", flush=True)
    print("-" * 48, flush=True)
    try:
        import transformers
        transformers.logging.set_verbosity_info()
        transformers.logging.enable_progress_bar()
    except Exception:
        pass
    try:
        capture_ocr.get_pipeline()
        print("-" * 48, flush=True)
        print("所有模型載入完成", flush=True)
        sys.exit(0)
    except Exception:
        print("-" * 48, flush=True)
        print("模型載入失敗：", flush=True)
        traceback.print_exc()
        sys.exit(1)


# ── FastAPI 應用 ────────────────────────────────────────────────────────

def _verify_key(x_api_key: Annotated[str | None, Header()] = None):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


def _preload():
    global _models_ready, _models_error
    try:
        capture_ocr.get_pipeline()
        _models_ready = True
        print("[INFO] OCR 模型預載完成", flush=True)
    except Exception:
        _models_error = traceback.format_exc()
        print(f"[ERROR] OCR 模型預載失敗：\n{_models_error}", flush=True)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_preload, daemon=True, name="preload").start()
    yield


app = FastAPI(title="OCR API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health(x_api_key: Annotated[str | None, Header()] = None):
    _verify_key(x_api_key)
    return {
        "status": 1,
        "models_ready": _models_ready,
        "models_error": _models_error,
    }


@app.post("/scan")
def scan(x_api_key: Annotated[str | None, Header()] = None):
    """
    觸發單次 OCR 辨識。

    回傳格式：
      {"status": 1,  "results": {"MFG": "...", "LOT": "...", ...}}
      {"status": -1, "message": "無瓶子"}
      {"status": 0,  "message": "培養瓶裝反了"}
      {"status": -2, "message": "錯誤訊息"}
    """
    _verify_key(x_api_key)

    if not _models_ready:
        if _models_error:
            return JSONResponse(
                status_code=500,
                content={"status": -2, "message": f"模型載入失敗：{_models_error}"},
            )
        return JSONResponse(
            status_code=503,
            content={"status": -2, "message": "OCR 模型仍在載入中，請稍後再試"},
        )

    result = capture_ocr.scan_once_result()
    return result


# ── 主程式 ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # 模型預下載模式（由 patch.exe 呼叫）
        if "--preload-models" in sys.argv:
            _preload_models_and_exit()

        # 模型未下載時僅警告，不強制退出
        ok, missing = _check_models_present()
        if not ok:
            print(f"[警告] 以下模型尚未下載：{', '.join(missing)}")
            print("建議先執行 patch.exe 下載模型（/scan 呼叫前需完成）")
            print()

        # 讀設定、選 port
        cfg = load_config()
        preferred_port: int = cfg.get("port", 8000)
        port = _find_port(preferred_port)
        if port != preferred_port:
            print(f"[警告] Port {preferred_port} 已被占用，改用 Port {port}")

        print("=" * 48)
        print("  OCR API Server v1.0")
        print("=" * 48)
        print(f"  伺服器：http://0.0.0.0:{port}")
        print(f"  API 文件：http://localhost:{port}/docs")
        print(f"  API Key：{API_KEY}")
        print("  關閉此視窗即可停止伺服器")
        print("=" * 48)

        uvicorn.run(app, host="0.0.0.0", port=port)

    except Exception:
        traceback.print_exc()
        input("\n[錯誤] 發生例外，請截圖後按 Enter 關閉...")
