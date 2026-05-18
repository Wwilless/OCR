"""OCR 影像擷取腳本 — ENTER: 辨識  P: 框選ROI  ESC: 離開"""

import sys, io, ctypes, os
ctypes.windll.kernel32.SetConsoleOutputCP(65001)
ctypes.windll.kernel32.SetConsoleCP(65001)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)

import re
import time
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from PIL import Image
import torch

# 攝影機解析度 645×560，所有框座標格式：x, y, w, h
ROI               = (97,  67, 424, 371)   # 綠框：OCR 辨識範圍，裁切後送入三引擎
BLUE_ROI          = (84,  13, 427,  50)   # 藍框：瓶口方向偵測，暗色占比高 → 方向正確
RED_ROI           = (526,  8,  99, 102)   # 紅框：有無瓶子偵測，暗色占比低 → 有瓶子
CHECK_BLUE_THRESH = 50                    # 藍框閾值
CHECK_BLUE_RATIO  = 0.50                  # 藍框比例門檻
CHECK_RED_THRESH  = 20                    # 紅框閾值
CHECK_RED_RATIO   = 0.20                  # 紅框比例門檻
CHECK_RED_INTERVAL = 2.0                  # 紅框偵測間隔
CHECK_RED_DEBOUNCE = 1.0                  # 等待時間
AUTO_OCR          = True                  # 紅框自動觸發辨識
BINARIZE_THRESH   = 140    # 灰階二值化閾值
CLOSE_KERNEL_SIZE = 3      # 形態學閉運算核大小（填補字元裂縫）
CLOSE_ITERATIONS  = 1      # 閉運算次數
BLOCK_BORDER      = 10     # 送入 TrOCR/Paddle 前每個字塊的額外邊距（px）
PADDLE_CLAHE_CLIP = 2.0
PADDLE_CLAHE_TILE = (8, 8)
ROW_FORMATS       = ['digits_only', 'alphanumeric']  # 各行格式：第0行純數字，第1行英數
ROW_LABELS        = ['MFG', 'LOT', 'MFD']            # 各行對應標籤

_CLAHE_GRAY  = cv2.createCLAHE(clipLimit=2.0,               tileGridSize=(8, 8))
_CLAHE_COLOR = cv2.createCLAHE(clipLimit=PADDLE_CLAHE_CLIP, tileGridSize=PADDLE_CLAHE_TILE)

_PREFIX_FIXES     = str.maketrans('15826', 'ISBZG')   
_DIGIT_FIXES      = str.maketrans('OISBZ',  '01582')
MONTH_LETTERS     = tuple('ABCDEFGHIJKL')
_KNOWN_PREFIXES   = frozenset({'MFG', 'MFD'})


# ── 影像前處理 ─────────────────────────────────────────────────────────

def crop_by_roi(image: np.ndarray, rx: int, ry: int, rw: int, rh: int) -> np.ndarray:
    # 依 ROI 座標安全裁切，超出邊界自動截斷
    ih, iw = image.shape[:2]
    x1, y1 = min(rx, iw), min(ry, ih)
    return image[y1:min(ry+rh, ih), x1:min(rx+rw, iw)]


def to_gray(image: np.ndarray) -> np.ndarray:
    # 灰階化 → CLAHE 對比增強 → 固定閾值二值化（供 EasyOCR / TrOCR）
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = _CLAHE_GRAY.apply(gray)
    _, binary = cv2.threshold(enhanced, BINARIZE_THRESH, 255, cv2.THRESH_BINARY)
    return binary


def apply_clahe_color(image: np.ndarray) -> np.ndarray:
    # LAB 色彩空間 CLAHE 對比增強，保留色彩（供 PaddleOCR）
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _CLAHE_COLOR.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def close_gaps(binary: np.ndarray) -> np.ndarray:
    # 形態學閉運算：填補字元筆劃裂縫，改善辨識率
    inv    = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    return cv2.bitwise_not(cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS))



# ── 偵測框工具 ─────────────────────────────────────────────────────────

def merge_same_row_rects(  # 將同一行的偵測框合併為單一行框
    rects: list[tuple[int,int,int,int]],
    texts: list[str],
    cy_ratio: float = 0.6,
) -> tuple[list, list]:
    if not rects:
        return rects, texts
    items   = sorted(zip(rects, texts), key=lambda p: p[0][1] + p[0][3] / 2)
    groups  = []
    current = [items[0]]
    for item in items[1:]:
        x, y, w, h = item[0]
        cy  = y + h / 2
        gy1 = min(r[1]        for r, _ in current)
        gy2 = max(r[1] + r[3] for r, _ in current)
        if abs(cy - (gy1+gy2)/2) < max(gy2-gy1, h) * cy_ratio:
            current.append(item)
        else:
            groups.append(current)
            current = [item]
    groups.append(current)
    result_rects, result_texts = [], []
    for group in groups:
        group.sort(key=lambda p: p[0][0])
        xs  = [r[0]       for r, _ in group]
        ys  = [r[1]       for r, _ in group]
        x2s = [r[0]+r[2]  for r, _ in group]
        y2s = [r[1]+r[3]  for r, _ in group]
        result_rects.append((min(xs), min(ys), max(x2s)-min(xs), max(y2s)-min(ys)))
        result_texts.append(" ".join(t for _, t in group))
    pairs = sorted(zip(result_rects, result_texts), key=lambda p: (p[0][1], p[0][0]))
    return [p[0] for p in pairs], [p[1] for p in pairs]



# ── 投票與格式驗證 ─────────────────────────────────────────────────────

def normalize_result(text: str, fmt: str) -> str:  # 統一格式為 "前綴:內容"，依 fmt 過濾字元
    t      = re.sub(r'\s', '', text).upper()
    prefix = t[:3] if len(t) >= 3 else t.ljust(3, '?')
    rest   = t[3:] if len(t) > 3  else ''
    content = rest[1:] if rest and rest[0] in '.:;,' else rest
    # 保留 '0' 供後續 O/D 判斷，其餘非英數字元轉 '?'
    prefix  = re.sub(r'[^A-Z0-9]', '?', prefix.translate(_PREFIX_FIXES))
    if fmt == 'digits_only':
        content = re.sub(r'[^0-9]', '', content.translate(_DIGIT_FIXES))
        content = content[-8:] if len(content) > 8 else content
    else:
        content = re.sub(r'[^A-Z0-9]', '', content)
    return prefix + ':' + content


def vote_chars(strings: list[str]) -> str:  # 三引擎結果逐字元多數決投票
    def _vote(candidates: list[str]) -> str:
        result = []
        for i in range(max(len(s) for s in candidates)):
            pool = [s[i] for s in candidates if i < len(s)]
            result.append(Counter(pool).most_common(1)[0][0])
        return ''.join(result)
    prefixes = [s.split(':')[0]                     for s in strings]
    contents = [s.split(':')[1] if ':' in s else '' for s in strings]
    return _vote(prefixes) + ':' + _vote(contents)


def _fix_mfg_content(content: str) -> str:
    d = re.sub(r'[^0-9]', '', content)[:8].ljust(8, '0')
    mm, dd = int(d[4:6]), int(d[6:8])
    if not (1 <= mm <= 12): print(f"  ⚠ MFG 月份異常: {d[4:6]}")
    if not (1 <= dd <= 31): print(f"  ⚠ MFG 日期異常: {d[6:8]}")
    return d


def _fix_lot_content(content: str, fallback_contents: list[str]) -> str:
    c       = re.sub(r'[^A-Z0-9]', '', content).ljust(6, '0')[:6]
    yy, x, nnn = c[0:2], c[2], c[3:6]
    if x not in MONTH_LETTERS:
        for fb in fallback_contents:
            fb_c = re.sub(r'[^A-Z0-9]', '', fb)
            if len(fb_c) > 2 and fb_c[2] in MONTH_LETTERS:
                print(f"  ⚠ LOT 月份字母修正: {x} → {fb_c[2]}")
                x = fb_c[2]
                break
        else:
            print(f"  ⚠ LOT 月份字母無法修正（三引擎皆異常）: {x}")
    if int(nnn) == 0:
        print(f"  ⚠ LOT 流水號異常: {nnn}")
    return yy + x + nnn


def _resolve_prefix(norms: tuple[str, str, str]) -> str:
    """對三引擎前綴與 _KNOWN_PREFIXES 做逐字元相似度加總，選分數最高的候選。"""
    def _char_sim(a: str, b: str) -> int:
        return sum(1 for x, y in zip(a, b) if x == y)

    engine_prefixes = [norm.split(':', 1)[0] if ':' in norm else '' for norm in norms]
    candidates = sorted(_KNOWN_PREFIXES)
    scores = {c: sum(_char_sim(p, c) for p in engine_prefixes) for c in candidates}
    best = max(candidates, key=lambda c: scores[c])

    return best


def apply_domain_rules(
    voted_labels: list[str],
    norm_per_row: list[tuple[str,str,str]],
) -> list[str]:
    result = list(voted_labels)
    for i, label in enumerate(result):
        if ':' not in label:
            continue
        prefix, content = label.split(':', 1)
        if i < len(norm_per_row):
            prefix = _resolve_prefix(norm_per_row[i])
        result[i] = prefix + ':' + content
    if result and ':' in result[0]:
        p, c = result[0].split(':', 1)
        result[0] = p + ':' + _fix_mfg_content(c)
    if len(result) > 1 and ':' in result[1] and len(norm_per_row) > 1:
        p, c = result[1].split(':', 1)
        fallbacks = [n.split(':', 1)[1] if ':' in n else '' for n in norm_per_row[1]]
        result[1] = p + ':' + _fix_lot_content(c, fallbacks)
    if len(result) >= 2:
        mfg_c = result[0].split(':', 1)[1] if ':' in result[0] else ''
        lot_c = result[1].split(':', 1)[1] if ':' in result[1] else ''
        if len(mfg_c) >= 4 and len(lot_c) >= 2 and mfg_c[2:4] != lot_c[0:2]:
            print(f"  ⚠ 年份不一致: MFG年={mfg_c[:4]}, LOT年份碼={lot_c[:2]}")
    return result


# ── OCR 引擎 ──────────────────────────────────────────────────────────

class EasyOCREngine:  # 通用文字偵測引擎，負責找出文字位置（rects）與初步辨識
    def __init__(self):
        self._model = None  # easyocr.Reader | None

    def load(self):
        if self._model is None:
            import logging, easyocr
            logging.getLogger('easyocr').setLevel(logging.ERROR)
            self._model = easyocr.Reader(["en"], gpu=False, verbose=False)

    def __call__(self, processed: np.ndarray) -> tuple[list, list]:
        self.load()
        inp = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR) if len(processed.shape) == 2 else processed
        rects, texts = [], []
        for (bbox, text, _prob) in (self._model.readtext(inp) or []):
            pts = np.array(bbox, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            rects.append((x, y, w, h))
            texts.append(text)
        pairs = sorted(zip(rects, texts), key=lambda p: (p[0][1], p[0][0]))
        return merge_same_row_rects([p[0] for p in pairs], [p[1] for p in pairs])


class TrOCREngine:  # 微軟 Transformer OCR，依 EasyOCR 框位逐塊精細辨識印刷體
    def __init__(self):
        self._processor = None  # (DeiTImageProcessor, AutoTokenizer)
        self._model     = None  # VisionEncoderDecoderModel | None

    def load(self):
        if self._processor is None:
            # transformers 4.45+ 與 torchvision 0.20+ 有相容性問題，繞過 TrOCRProcessor
            from transformers import DeiTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
            import transformers
            transformers.logging.set_verbosity_error()
            transformers.logging.disable_progress_bar()
            _model_id = "microsoft/trocr-small-printed"
            # processor 與 tokenizer 可並行載入
            with ThreadPoolExecutor(max_workers=2) as loader:
                fut_ip = loader.submit(DeiTImageProcessor.from_pretrained, _model_id)
                fut_tk = loader.submit(AutoTokenizer.from_pretrained,      _model_id, use_fast=False)
                image_processor = fut_ip.result()
                tokenizer       = fut_tk.result()
            self._processor = (image_processor, tokenizer)
            self._model = VisionEncoderDecoderModel.from_pretrained(_model_id)
            self._model.eval()

    def __call__(self, cropped: np.ndarray, rects: list[tuple[int,int,int,int]]) -> list[str]:
        self.load()
        image_processor, tokenizer = self._processor
        ih, iw = cropped.shape[:2]
        texts  = [""] * len(rects)
        pils, valid_idx = [], []
        for i, (x, y, bw, bh) in enumerate(rects):
            x1, x2 = max(0, x-BLOCK_BORDER), min(iw, x+bw+BLOCK_BORDER)
            y1, y2 = max(0, y-BLOCK_BORDER), min(ih, y+bh+BLOCK_BORDER)
            region = cropped[y1:y2, x1:x2]
            if region.size == 0:
                continue
            pils.append(Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB)))
            valid_idx.append(i)
        if pils:
            pixel_values = image_processor(images=pils, return_tensors="pt").pixel_values
            with torch.inference_mode():
                ids = self._model.generate(pixel_values)
            for out_i, orig_i in enumerate(valid_idx):
                texts[orig_i] = tokenizer.decode(ids[out_i], skip_special_tokens=True)
        return texts


class PaddleOCREngine:  # 百度 PaddleOCR 英文引擎，依 EasyOCR 框位逐塊辨識
    def __init__(self):
        self._model = None  # PaddleOCR | None

    def load(self):
        if self._model is None:
            import warnings, logging, os
            # 環境變數與 logging 均須在 import paddle 前設定
            os.environ.setdefault('GLOG_minloglevel',       '3')
            os.environ.setdefault('GLOG_logtostderr',       '0')
            os.environ.setdefault('FLAGS_call_stack_level', '0')
            warnings.filterwarnings('ignore')
            for _lg in ('paddlex', 'paddleocr', 'paddle', 'ppocr', 'root'):
                logging.getLogger(_lg).setLevel(logging.ERROR)
            from paddleocr import PaddleOCR
            self._model = PaddleOCR(
                lang="en",
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
                enable_mkldnn=False,
            )

    def __call__(self, cropped: np.ndarray, rects: list[tuple[int,int,int,int]]) -> list[str]:
        self.load()
        ih, iw = cropped.shape[:2]
        texts  = []
        for (x, y, bw, bh) in rects:
            x1, x2 = max(0, x-BLOCK_BORDER), min(iw, x+bw+BLOCK_BORDER)
            y1, y2 = max(0, y-BLOCK_BORDER), min(ih, y+bh+BLOCK_BORDER)
            region = cropped[y1:y2, x1:x2]
            if region.size == 0:
                texts.append("")
                continue
            results  = self._model.predict(region)
            rec_list = results[0].get("rec_texts", []) if results else []
            texts.append(" ".join(t for t in rec_list if t))
        return texts


# ── OCR 流程管理 ───────────────────────────────────────────────────────

class OCRPipeline:  # 統籌三引擎：EasyOCR 偵測框 → TrOCR+Paddle 並行辨識 → 投票+驗證輸出
    def __init__(self):
        self.easy   = EasyOCREngine()
        self.trocr  = TrOCREngine()
        self.paddle = PaddleOCREngine()
        self._pool  = ThreadPoolExecutor(max_workers=3)

    def preload(self):
        fut_e = self._pool.submit(self.easy.load)
        fut_t = self._pool.submit(self.trocr.load)
        fut_p = self._pool.submit(self.paddle.load)
        fut_e.result()
        fut_t.result()
        fut_p.result()

    def scan(self, frame: np.ndarray, roi: tuple[int,int,int,int]) -> list[str]:
        rx, ry, rw, rh = roi
        cropped = crop_by_roi(frame, rx, ry, rw, rh)
        binary = to_gray(cropped)
        processed = close_gaps(binary)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        paddle_input = apply_clahe_color(cropped)

        def _split_label(text):
            t = re.sub(r'\s', '', text).upper()
            label   = t[:3] if len(t) >= 3 else t.ljust(3)
            rest    = t[3:] if len(t) > 3 else ''
            content = rest[1:] if rest and rest[0] in '.:;,' else rest
            return label, content

        # Step 1: EasyOCR
        rects, texts = self.easy(processed)

        # Step 2+3: TrOCR + PaddleOCR 並行
        fut_t = self._pool.submit(self.trocr,  processed_bgr, rects)
        fut_p = self._pool.submit(self.paddle, paddle_input,  rects)
        trocr_texts  = fut_t.result()
        paddle_texts = fut_p.result()

        voted_labels, norm_per_row = [], []
        for i, (raw_e, raw_t, raw_p) in enumerate(zip(texts, trocr_texts, paddle_texts)):
            fmt = ROW_FORMATS[i] if i < len(ROW_FORMATS) else 'alphanumeric'
            norm_e, norm_t, norm_p = [normalize_result(r, fmt) for r in (raw_e, raw_t, raw_p)]
            voted = vote_chars([norm_e, norm_t, norm_p])
            voted_labels.append(voted)
            norm_per_row.append((norm_e, norm_t, norm_p))

        voted_labels = apply_domain_rules(voted_labels, norm_per_row)
        final_labels = []
        for i, label in enumerate(voted_labels):
            pfx, cnt = label.split(':', 1) if ':' in label else ('', label)
            if i > 0 and i < len(ROW_LABELS):
                pfx = ROW_LABELS[i]
            final_labels.append(f"{pfx}:{cnt}")
            print(f"{pfx} : {cnt}")

        return final_labels


# ── 瓶口方向檢查 ───────────────────────────────────────────────────────

def get_dark_ratio(frame: np.ndarray) -> float:
    bx, by, bw, bh = BLUE_ROI
    region = frame[by:by+bh, bx:bx+bw]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray < CHECK_BLUE_THRESH) / gray.size)


def check_bottle_orientation(frame: np.ndarray) -> bool:
    return get_dark_ratio(frame) >= CHECK_BLUE_RATIO


def get_red_dark_ratio(frame: np.ndarray) -> float:
    rx, ry, rw, rh = RED_ROI
    region = frame[ry:ry+rh, rx:rx+rw]
    if region.size == 0:
        return 1.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray < CHECK_RED_THRESH) / gray.size)


# ── 主程式 ─────────────────────────────────────────────────────────────

def main():
    rx, ry, rw, rh = ROI
    print(f"ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    cam_idx = 1
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("錯誤：找不到可用攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  645)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 560)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    print("操作說明:  ENTER -> 辨識  ESC -> 離開")

    window_name = "OCR"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 645, 560)

    show_overlay          = False  # M 鍵切換：顯示/隱藏綠框、藍框、紅框
    pipeline              = OCRPipeline()
    import threading
    preload_thread        = threading.Thread(target=pipeline.preload, daemon=True, name="preload")
    preload_thread.start()
    print("OCR 載入中，請稍候...")
    preload_thread.join()
    print("OCR 載入完成，開始辨識")
    prev_has_bottle       = False  # True 在掃描或方向錯誤後設定，瓶子離開才重置
    last_displayed_bottle = False  # 僅用於「有瓶子/沒瓶子」印出去重
    last_red_check        = 0.0   # 上次紅框偵測時間
    bottle_trigger_at     = 0.0   # 首次偵測到有瓶子的時間（debounce 起點）
    status_lines          = ["Standby"]  # 左上角顯示文字

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            print("無法讀取畫面，重新連接...")
            cap.release()
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            continue

        # frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        display = frame.copy()
        if show_overlay:
            cv2.rectangle(display, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            bx, by, bw, bh = BLUE_ROI
            cv2.rectangle(display, (bx, by), (bx+bw, by+bh), (255, 0, 0), 2)
            ex, ey, ew, eh = RED_ROI
            cv2.rectangle(display, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
        for i, line in enumerate(status_lines):
            cv2.putText(display, line, (10, 30 + i * 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow(window_name, display)

        # ── 自動偵測：紅框邊緣觸發（用未畫框的原始 frame 偵測）──────────
        if time.time() - last_red_check >= CHECK_RED_INTERVAL:
            last_red_check = time.time()
            dark_ratio = get_red_dark_ratio(frame)
            has_bottle = dark_ratio < CHECK_RED_RATIO  # 暗色少 → 有瓶子

            if has_bottle != last_displayed_bottle:
                if not has_bottle:
                    print("無瓶子", flush=True)
                status_lines = ["Bottle detected" if has_bottle else "No bottle"]
                last_displayed_bottle = has_bottle
                if has_bottle:
                    bottle_trigger_at = time.time()  # 記錄變化時間點，開始 debounce

            if AUTO_OCR and has_bottle and not prev_has_bottle and time.time() - bottle_trigger_at >= CHECK_RED_DEBOUNCE:
                if not check_bottle_orientation(frame):
                    print("培養瓶裝反了", flush=True)
                    status_lines = ["Wrong orientation"]
                    prev_has_bottle = True
                else:
                    if preload_thread.is_alive():
                        print("OCR 載入中，請稍候...")
                        preload_thread.join()
                    print("辨識中...")
                    try:
                        results = pipeline.scan(frame, (rx, ry, rw, rh))
                        if results:
                            status_lines = [lbl.replace(':', ': ', 1) for lbl in results]
                    except Exception:
                        traceback.print_exc()
                    prev_has_bottle = True

            if not has_bottle:
                prev_has_bottle = False  # 瓶子離開，重置讓下次進入可觸發

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key in (ord('m'), ord('M')):
            show_overlay = not show_overlay

        elif key == 13:
            if not check_bottle_orientation(frame):
                print(":培養瓶裝反了")
                status_lines = ["Blue: Wrong orientation"]
            else:
                if preload_thread.is_alive():
                    print("OCR 引擎載入中，請稍候...")
                    preload_thread.join()
                print("辨識中...")
                try:
                    results = pipeline.scan(frame, (rx, ry, rw, rh))
                    if results:
                        status_lines = [f"{ROW_LABELS[i] if i < len(ROW_LABELS) else str(i)}: {lbl}" for i, lbl in enumerate(results)]
                except Exception:
                
                    traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    print("程式結束")


if __name__ == "__main__":
    main()

