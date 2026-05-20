"""OCR API Patch — 攝影機設定 + 模型下載（第一次使用時執行）"""

import subprocess
import sys
from pathlib import Path

from config_manager import load_config, save_config


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _server_cmd() -> list[str]:
    if getattr(sys, "frozen", False):
        return [str(_base_dir() / "fastapi_server.exe"), "--preload-models"]
    return [sys.executable, str(_base_dir() / "fastapi_server.py"), "--preload-models"]


def _check_models() -> dict[str, bool]:
    home = Path.home()
    hub = home / ".cache" / "huggingface" / "hub"
    return {
        "EasyOCR":   (home / ".EasyOCR" / "model").exists(),
        "TrOCR":     hub.exists() and any(hub.glob("models--microsoft--trocr-small-printed*")),
        "PaddleOCR": (home / ".paddleocr").exists(),
    }


def _setup_camera(cfg: dict) -> dict:
    print(f"[攝影機設定] 目前索引: {cfg['camera_index']}")
    raw = input("  輸入新索引後按 Enter 修改，直接按 Enter 略過: ").strip()
    if raw.isdigit():
        cfg["camera_index"] = int(raw)
        print(f"  攝影機索引已設定為: {cfg['camera_index']}")
    return cfg


def _download_models() -> bool:
    cmd = _server_cmd()
    server_path = Path(cmd[0])

    if not server_path.exists():
        print(f"[錯誤] 找不到 {server_path.name}")
        print("請確認 fastapi_server.exe 與 patch.exe 在同一資料夾")
        return False

    print("  正在啟動下載程序，請勿關閉視窗...", flush=True)
    print("-" * 48, flush=True)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
        proc.wait()
        print("-" * 48, flush=True)
        return proc.returncode == 0
    except Exception as e:
        print(f"[錯誤] 執行失敗：{e}", flush=True)
        return False


def main() -> None:
    print("=" * 48)
    print("  OCR API Patch v1.0")
    print("=" * 48)
    print()

    # 攝影機設定
    cfg = _setup_camera(load_config())
    save_config(cfg)
    print()

    # 模型狀態檢查
    print("[模型檢查]")
    models = _check_models()
    for name, ok in models.items():
        status = "OK  已存在" if ok else "!!  尚未下載"
        print(f"  {name:<12} {status}")
    print()

    missing = [name for name, ok in models.items() if not ok]
    if missing:
        print(f"[模型下載] 需要下載：{', '.join(missing)}")
        success = _download_models()
        if success:
            print("模型下載完成！")
        else:
            print("[警告] 下載過程出現錯誤，請重新執行 patch.exe 或檢查網路連線")
    else:
        print("所有模型已就緒，無需下載")

    print()
    print("設定完成！請執行 fastapi_server.exe 啟動伺服器")
    print()
    input("按 Enter 關閉...")


if __name__ == "__main__":
    main()
