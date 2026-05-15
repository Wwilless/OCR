"""OCR 影像擷取腳本 — ENTER: 辨識  P: 框選ROI  ESC: 離開"""

import re
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy as np

ROI               = (454, 168, 960, 838)  # x, y, w, h
CHECK_ABOVE_H     = 150                    # 藍框高度 (px)
CHECK_BLUE_THRESH = 50                     # 藍框：暗色像素閾值 (0~255)
CHECK_BLUE_RATIO  = 0.50                   # 藍框：暗色比例門檻
CHECK_RED_H       = 100                    # 紅框寬度 (px)
CHECK_RED_V       = 150                    # 紅框高度 (px)
CHECK_RED_THRESH  = 20                     # 紅框：暗色像素閾值 (0~255)
CHECK_RED_RATIO   = 0.20                   # 紅框：暗色比例門檻
BINARIZE_THRESH   = 140
CLOSE_KERNEL_SIZE = 3
CLOSE_ITERATIONS  = 1
BLOCK_BORDER      = 10
PADDLE_CLAHE_CLIP = 2.0
PADDLE_CLAHE_TILE = (8, 8)
ROW_FORMATS       = ['digits_only', 'alphanumeric']

_PREFIX_FIXES = str.maketrans('015826', 'OISBZG')
_DIGIT_FIXES  = str.maketrans('OISBZ',  '01582')
MONTH_LETTERS = tuple('ABCDEFGHIJKL')

# active:bool  start/end:tuple[int,int]|None  frame:np.ndarray|None
_sel: dict = {"active": False, "start": None, "end": None, "frame": None}


# ── 滑鼠回呼 ──────────────────────────────────────────────────────────

def _mouse_cb(event, x, y, *_):
    if not _sel["active"]:
        return
    if event == cv2.EVENT_LBUTTONDOWN:
        _sel["start"] = _sel["end"] = (x, y)
    elif _sel["start"] and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
        _sel["end"] = (x, y)


# ── 影像前處理 ─────────────────────────────────────────────────────────

def crop_by_roi(image: np.ndarray, rx: int, ry: int, rw: int, rh: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    x1, y1 = min(rx, iw), min(ry, ih)
    return image[y1:min(ry+rh, ih), x1:min(rx+rw, iw)]


def to_gray(image: np.ndarray) -> np.ndarray:
    gray     = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    _, binary = cv2.threshold(enhanced, BINARIZE_THRESH, 255, cv2.THRESH_BINARY)
    return binary


def apply_clahe_color(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=PADDLE_CLAHE_CLIP, tileGridSize=PADDLE_CLAHE_TILE).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def close_gaps(binary: np.ndarray) -> np.ndarray:
    inv    = cv2.bitwise_not(binary)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE))
    return cv2.bitwise_not(cv2.morphologyEx(inv, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITERATIONS))


# ── 偵測框工具 ─────────────────────────────────────────────────────────

def merge_same_row_rects(
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

def normalize_result(text: str, fmt: str) -> str:
    t      = re.sub(r'\s', '', text).upper()
    prefix = t[:3] if len(t) >= 3 else t.ljust(3, '?')
    rest   = t[3:] if len(t) > 3  else ''
    # 冒號位置可能被誤讀為 . ; ,
    content = rest[1:] if rest and rest[0] in '.:;,' else rest
    prefix  = re.sub(r'[^A-Z]', '?', prefix.translate(_PREFIX_FIXES))
    if fmt == 'digits_only':
        content = re.sub(r'[^0-9]', '', content.translate(_DIGIT_FIXES))
        content = content[-8:] if len(content) > 8 else content
    else:
        content = re.sub(r'[^A-Z0-9]', '', content)
    return prefix + ':' + content


def vote_chars(strings: list[str]) -> str:
    from collections import Counter
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


def apply_domain_rules(
    voted_labels: list[str],
    norm_per_row: list[tuple[str,str,str]],
) -> list[str]:
    result = list(voted_labels)
    for i, label in enumerate(result):
        if ':' not in label:
            continue
        prefix, content = label.split(':', 1)
        if not re.fullmatch(r'[A-Z]{3}', prefix):
            if i < len(norm_per_row):
                for norm in norm_per_row[i]:
                    fb = norm.split(':', 1)[0] if ':' in norm else ''
                    if re.fullmatch(r'[A-Z]{3}', fb):
                        print(f"  ⚠ 前綴修正: {prefix} → {fb}")
                        prefix = fb
                        break
                else:
                    print(f"  ⚠ 前綴異常無法修正: {prefix}")
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

class EasyOCREngine:
    def __init__(self):
        self._model = None  # easyocr.Reader | None

    def load(self):
        if self._model is None:
            print("Thread 執行同步中 (EasyOCR)...")
            import easyocr
            self._model = easyocr.Reader(["en"], gpu=False)
            print("Thread 執行同步完成 (EasyOCR)")

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


class TrOCREngine:
    def __init__(self):
        self._processor = None  # (DeiTImageProcessor, AutoTokenizer)
        self._model     = None  # VisionEncoderDecoderModel | None

    def load(self):
        if self._processor is None:
            print("Thread 執行同步中 (TrOCR)...")
            # transformers 4.45+ 與 torchvision 0.20+ 有相容性問題，繞過 TrOCRProcessor
            from transformers import DeiTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel
            # processor 與 tokenizer 可並行載入
            with ThreadPoolExecutor(max_workers=2) as loader:
                fut_ip = loader.submit(DeiTImageProcessor.from_pretrained, "microsoft/trocr-small-printed", local_files_only=True)
                fut_tk = loader.submit(AutoTokenizer.from_pretrained,      "microsoft/trocr-small-printed", use_fast=False, local_files_only=True)
                image_processor = fut_ip.result()
                tokenizer       = fut_tk.result()
            self._processor = (image_processor, tokenizer)
            self._model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-printed", local_files_only=True)
            self._model.eval()
            print("Thread 執行同步完成 (TrOCR)")

    def __call__(self, cropped: np.ndarray, rects: list[tuple[int,int,int,int]]) -> list[str]:
        from PIL import Image
        import torch
        self.load()
        image_processor, tokenizer = self._processor
        ih, iw = cropped.shape[:2]
        texts  = []
        for (x, y, bw, bh) in rects:
            x1, x2 = max(0, x-BLOCK_BORDER), min(iw, x+bw+BLOCK_BORDER)
            y1, y2 = max(0, y-BLOCK_BORDER), min(ih, y+bh+BLOCK_BORDER)
            region = cropped[y1:y2, x1:x2]
            if region.size == 0:
                texts.append("")
                continue
            pil = Image.fromarray(cv2.cvtColor(region, cv2.COLOR_BGR2RGB))
            pixel_values = image_processor(images=pil, return_tensors="pt").pixel_values
            with torch.no_grad():
                ids = self._model.generate(pixel_values)
            texts.append(tokenizer.decode(ids[0], skip_special_tokens=True))
        return texts


class PaddleOCREngine:
    def __init__(self):
        self._model = None  # PaddleOCR | None

    def load(self):
        if self._model is None:
            import warnings, logging, os
            warnings.filterwarnings('ignore', category=UserWarning)
            from paddleocr import PaddleOCR
            logging.getLogger('paddlex').setLevel(logging.ERROR)
            logging.getLogger('paddleocr').setLevel(logging.ERROR)
            print("Thread 執行同步中 (PaddleOCR)...")
            # C++ 層（glog/oneDNN）直接寫 fd2，dup2 重導向至 /dev/null 遮蔽雜訊
            _devnull = os.open(os.devnull, os.O_WRONLY)
            _saved   = os.dup(2)
            os.dup2(_devnull, 2)
            os.close(_devnull)
            try:
                self._model = PaddleOCR(
                    lang="en",
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                    enable_mkldnn=False,
                )
            finally:
                os.dup2(_saved, 2)
                os.close(_saved)
            print("Thread 執行同步完成 (PaddleOCR)")

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

class OCRPipeline:
    def __init__(self):
        self.easy   = EasyOCREngine()
        self.trocr  = TrOCREngine()
        self.paddle = PaddleOCREngine()
        self._pool  = ThreadPoolExecutor(max_workers=3)

    def preload(self):
        # EasyOCR 與 TrOCR 可並行載入；PaddleOCR 需單獨載入（os.dup2 stderr 重導向不可並行）
        fut_e = self._pool.submit(self.easy.load)
        fut_t = self._pool.submit(self.trocr.load)
        fut_e.result()
        fut_t.result()
        self.paddle.load()

    def scan(self, frame: np.ndarray, roi: tuple[int,int,int,int]) -> list[str]:
        t_total = time.time()
        rx, ry, rw, rh = roi
        cropped = crop_by_roi(frame, rx, ry, rw, rh)
        binary = to_gray(cropped)
        processed = close_gaps(binary)
        processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
        paddle_input = apply_clahe_color(cropped)

        # Step 1: EasyOCR
        rects, texts = self.easy(processed)
        easy_labels = [s.upper() for s in texts]
        for label in easy_labels: print(f"easyOCR   : {label}")

        # Step 2+3: TrOCR + PaddleOCR 並行
        fut_t = self._pool.submit(self.trocr,  processed_bgr, rects)
        fut_p = self._pool.submit(self.paddle, paddle_input,  rects)
        trocr_texts  = fut_t.result()
        paddle_texts = fut_p.result()
        trocr_labels  = [re.sub(r'[^A-Z0-9]', '', s.upper()) for s in trocr_texts]
        paddle_labels = [re.sub(r'[^A-Z0-9]', '', s.upper()) for s in paddle_texts]
        for label in trocr_labels:  print(f"TrOCR     : {label}")
        for label in paddle_labels: print(f"PaddleOCR : {label}")

        voted_labels, norm_per_row = [], []
        for i, (raw_e, raw_t, raw_p) in enumerate(zip(texts, trocr_texts, paddle_texts)):
            fmt = ROW_FORMATS[i] if i < len(ROW_FORMATS) else 'alphanumeric'
            norm_e, norm_t, norm_p = [normalize_result(r, fmt) for r in (raw_e, raw_t, raw_p)]
            voted = vote_chars([norm_e, norm_t, norm_p])
            voted_labels.append(voted)
            norm_per_row.append((norm_e, norm_t, norm_p))
            print(f"投票結果  : {voted}  ({norm_e} | {norm_t} | {norm_p})")

        voted_labels = apply_domain_rules(voted_labels, norm_per_row)
        for label in voted_labels: print(f"驗證結果  : {label}")
        print(f"總耗時           : {time.time() - t_total:.2f} 秒")
        return voted_labels


# ── 瓶口方向檢查 ───────────────────────────────────────────────────────

def get_dark_ratio(frame: np.ndarray, rx: int, ry: int, rw: int) -> float:
    y1 = max(0, ry - CHECK_ABOVE_H - 10)
    y2 = max(0, ry - 10)
    region = frame[y1:y2, rx:rx + rw]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray < CHECK_BLUE_THRESH) / gray.size)


def check_bottle_orientation(frame: np.ndarray, rx: int, ry: int, rw: int) -> bool:
    dark_ratio = get_dark_ratio(frame, rx, ry, rw)
    print(f"藍框暗色比例: {dark_ratio:.2%}（門檻 {CHECK_BLUE_RATIO:.0%}）")
    return dark_ratio >= CHECK_BLUE_RATIO


def get_red_ratio(frame: np.ndarray, rx: int, ry: int, rw: int) -> float:
    x1 = rx + rw + 5
    x2 = min(frame.shape[1], x1 + CHECK_RED_H)
    y1 = max(0, ry - CHECK_RED_V - 10)
    y2 = max(0, ry - 10)
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return 0.0
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    return float(np.sum(gray > CHECK_RED_THRESH) / gray.size)


# ── 主程式 ─────────────────────────────────────────────────────────────

def main():
    rx, ry, rw, rh = ROI
    print(f"ROI: x={rx}, y={ry}, w={rw}, h={rh}")

    cam_idx = 1
    cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("錯誤：找不到可用攝影機")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"攝影機 {cam_idx} 已開啟，原生解析度: {w}x{h}，顯示解析度: 1920x1080")
    print("操作說明:  ENTER -> 辨識  P -> 框選ROI  ESC -> 離開")

    DISPLAY_W, DISPLAY_H = 1920, 1080
    window_name = "OCR"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    cv2.setMouseCallback(window_name, _mouse_cb)

    pipeline         = OCRPipeline()
    preload_thread   = None
    prev_has_bottle  = False

    while True:
        cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            print("無法讀取畫面，重新連接...")
            cap.release()
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            continue

        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        if _sel["active"]:
            display = _sel["frame"].copy()
            if _sel["start"] and _sel["end"]:
                cv2.rectangle(display, _sel["start"], _sel["end"], (0, 255, 255), 2)
            cv2.putText(display, "Drag to select ROI | P: Apply  O: Cancel",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
        else:
            cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), (0, 255, 0), 2)
            blue_y1 = max(0, ry - CHECK_ABOVE_H - 10)
            blue_y2 = ry - 10
            red_y1  = max(0, ry - CHECK_RED_V - 10)
            red_y2  = ry - 10
            cv2.rectangle(frame, (rx, blue_y1), (rx + rw, blue_y2), (255, 0, 0), 2)
            cv2.rectangle(frame, (rx + rw + 5, red_y1), (rx + rw + 5 + CHECK_RED_H, red_y2), (0, 0, 255), 2)
            cv2.putText(frame, f"Cam {cam_idx} | 自動偵測中  ENTER: 手動掃描  P: ROI  ESC: Quit",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, frame)

        # ── 自動偵測：紅框邊緣觸發 ──────────────────────────────────────
        if not _sel["active"]:
            has_bottle = get_red_ratio(frame, rx, ry, rw) < CHECK_RED_RATIO
            if has_bottle != prev_has_bottle:
                print("有瓶子" if has_bottle else "沒瓶子")
            if has_bottle and not prev_has_bottle:
                if not check_bottle_orientation(frame, rx, ry, rw):
                    print("錯誤：培養瓶裝反了")
                else:
                    if preload_thread is None:
                        import threading
                        preload_thread = threading.Thread(target=pipeline.preload, daemon=True, name="preload")
                        preload_thread.start()
                    if preload_thread.is_alive():
                        print("OCR 引擎載入中，請稍候...")
                        preload_thread.join()
                    print("辨識中...")
                    try:
                        pipeline.scan(frame, (rx, ry, rw, rh))
                    except Exception:
                        import traceback
                        traceback.print_exc()
            prev_has_bottle = has_bottle

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key in (ord('p'), ord('P')):
            if not _sel["active"]:
                _sel.update(active=True, start=None, end=None, frame=frame.copy())
                print("框選模式：拖曳滑鼠選取 ROI，再按 P 套用，O 取消")
            else:
                if _sel["start"] and _sel["end"]:
                    x1, y1 = _sel["start"]
                    x2, y2 = _sel["end"]
                    rx, ry = min(x1, x2), min(y1, y2)
                    rw, rh = abs(x2-x1), abs(y2-y1)
                    print(f"ROI = ({rx}, {ry}, {rw}, {rh})")
                _sel.update(active=False, start=None, end=None, frame=None)

        elif key in (ord('o'), ord('O')):
            _sel.update(active=False, start=None, end=None, frame=None)
            print("框選已取消")

        elif key == 13:
            if not check_bottle_orientation(frame, rx, ry, rw):
                print("錯誤：培養瓶裝反了")
            else:
                if preload_thread is None:
                    import threading
                    preload_thread = threading.Thread(target=pipeline.preload, daemon=True, name="preload")
                    preload_thread.start()
                if preload_thread.is_alive():
                    print("OCR 引擎載入中，請稍候...")
                    preload_thread.join()
                print("辨識中...")
                try:
                    pipeline.scan(frame, (rx, ry, rw, rh))
                except Exception:
                    import traceback
                    traceback.print_exc()

    cap.release()
    cv2.destroyAllWindows()
    print("程式結束")


if __name__ == "__main__":
    main()

