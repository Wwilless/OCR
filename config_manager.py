"""共用設定管理：讀寫 config.json，PyInstaller exe 與 .py 腳本均適用"""

import json
import sys
from pathlib import Path

_DEFAULTS: dict = {"camera_index": 1, "port": 8000}


def _base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent


def _config_path() -> Path:
    return _base_dir() / "config.json"


def load_config() -> dict:
    p = _config_path()
    if p.exists():
        try:
            return {**_DEFAULTS, **json.loads(p.read_text(encoding="utf-8"))}
        except Exception:
            pass
    return dict(_DEFAULTS)


def save_config(cfg: dict) -> None:
    _config_path().write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
