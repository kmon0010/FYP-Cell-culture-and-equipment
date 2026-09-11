"""
Depth + YOLO 3D Tracker GUI
----------------------------
PySide6 GUI that mirrors the structure of the existing E23 GUI:
  - QMainWindow → QTabWidget → "Live Feed" tab + "Review" tab
  - Live Feed:  RGB + depth side-by-side, model selector, source selector,
                recording controls, 3D detection log, terminal
  - Review:     Browse saved clips / CSVs, replay pre-recorded video with YOLO

Camera intrinsics — edit before running:
    FX, FY, CX, CY  (depth image resolution 320×240)

Dependencies:
    pip install PySide6 opencv-python ultralytics numpy requests
"""

import os, sys, csv, time, struct, threading
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2
import requests
import matplotlib.cm as cm
from ultralytics import YOLO

import track3d_core as core

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QGridLayout, QVBoxLayout, QHBoxLayout, QFrame,
    QLabel, QPushButton, QTextEdit, QComboBox,
    QFileDialog, QListWidget, QListWidgetItem, QToolBar, QToolButton,
    QSizePolicy, QScrollArea
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, QObject
from PySide6.QtGui import QImage, QPixmap, QFont, QColor

# ─────────────────────────────────────────────────────────────────────────────
# Camera / depth settings
# ─────────────────────────────────────────────────────────────────────────────
HOST = "192.168.233.1"
PORT = 80

FX, FY = 200.0, 200.0          # ← replace with your calibrated values
CX, CY = 160.0, 120.0

DEPTH_OLD_MIN, DEPTH_OLD_MAX = 16, 255
DEPTH_NEW_MIN, DEPTH_NEW_MAX = 150, 1500   # mm

RGB_W,   RGB_H   = 640, 480
DEPTH_W, DEPTH_H = 320, 240
SCALE_X, SCALE_Y = DEPTH_W / RGB_W, DEPTH_H / RGB_H

CLIP_DURATION_SEC = 5
ESTIMATED_FPS     = 10
OUTPUT_DIR        = "clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "detections_3d.csv")

# ─────────────────────────────────────────────────────────────────────────────
# Camera protocol helpers
# ─────────────────────────────────────────────────────────────────────────────
def frame_config_decode(raw): return struct.unpack("<BBBBBBBBi", raw)

def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255,
                         ir_mode=1, status_mode=2, status_mask=7,
                         rgb_mode=1, rgb_res=0, expose_time=0):
    return struct.pack("<BBBBBBBBi", trigger_mode, deep_mode, deep_shift,
                       ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)

def frame_payload_decode(frame_data, cfg):
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    payload = frame_data[8:]
    deepth_size = (320*240*2) >> cfg[1]
    depth_bytes = payload[:deepth_size] if deepth_size else None
    payload = payload[deepth_size:]
    ir_size = (320*240*2) >> cfg[3]
    payload = payload[ir_size:]
    status_size = (320*240//8)*(16 if cfg[4]==0 else 2 if cfg[4]==1 else 8 if cfg[4]==2 else 1)
    payload = payload[status_size:]
    rgb_size = len(payload)
    rgb_bytes = payload[:rgb_size] if rgb_size else None
    if rgb_bytes and cfg[6] == 1:
        jpeg = cv2.imdecode(np.frombuffer(rgb_bytes, "uint8", rgb_size), cv2.IMREAD_COLOR)
        rgb_bytes = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB).tobytes() if jpeg is not None else None
    return depth_bytes, rgb_bytes

def post_encode_config(config=frame_config_encode(), host=HOST, port=PORT):
    try:
        r = requests.post(f"http://{host}:{port}/set_cfg", config, timeout=5)
        return r.status_code == 200
    except Exception:
        return False

def get_frame_from_http(host=HOST, port=PORT):
    try:
        r = requests.get(f"http://{host}:{port}/getdeep", timeout=5)
        if r.status_code == 200:
            return r.content
    except Exception:
        pass
    return None

def decode_raw_frame(raw):
    cfg = frame_config_decode(raw[16:28])
    depth_bytes, rgb_bytes = frame_payload_decode(raw[28:], cfg)
    depth_mm = None
    if depth_bytes:
        dtype = "uint16" if cfg[1] == 0 else "uint8"
        raw_d = np.frombuffer(depth_bytes, dtype).reshape(240, 320).astype(float)
        depth_mm = ((raw_d - DEPTH_OLD_MIN)*(DEPTH_NEW_MAX - DEPTH_NEW_MIN) /
                    (DEPTH_OLD_MAX - DEPTH_OLD_MIN) + DEPTH_NEW_MIN).astype(np.float32)
    rgb = np.frombuffer(rgb_bytes, "uint8").reshape(480, 640, 3) if rgb_bytes else None
    return depth_mm, rgb

def depth_to_colormap_bgr(depth_mm):
    norm = np.clip((depth_mm - DEPTH_NEW_MIN)/(DEPTH_NEW_MAX - DEPTH_NEW_MIN), 0, 1)
    colored = (cm.jet_r(norm)[:, :, :3]*255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

def backproject(cx_rgb, cy_rgb, depth_mm):
    dcx = int(np.clip(cx_rgb*SCALE_X, 0, DEPTH_W-1))
    dcy = int(np.clip(cy_rgb*SCALE_Y, 0, DEPTH_H-1))
    Z = float(depth_mm[dcy, dcx])
    if Z <= 0:
        return None, dcx, dcy
    X = (dcx - CX)*Z/FX
    Y = (dcy - CY)*Z/FY
    return (X, Y, Z), dcx, dcy

# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────
_csv_lock = threading.Lock()

def init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","frame_id","track_id","class",
                                    "cx_px","cy_px","X_mm","Y_mm","Z_mm"])

def log_detection_csv(frame_id, track_id, class_name, cx, cy, X, Y, Z):
    ts = datetime.now().isoformat(timespec="milliseconds")
    with _csv_lock:
        with open(CSV_PATH, "a", newline="") as f:
            csv.writer(f).writerow([ts, frame_id, track_id, class_name,
                                    round(cx,1), round(cy,1),
                                    round(X,1), round(Y,1), round(Z,1)])

# ─────────────────────────────────────────────────────────────────────────────
# Clip recording
# ─────────────────────────────────────────────────────────────────────────────
def save_clip(rgb_frames, depth_color_frames, depth_raw_frames, ts):
    clip = os.path.join(OUTPUT_DIR, ts)
    fps  = ESTIMATED_FPS
    if rgb_frames:
        h, w = rgb_frames[0].shape[:2]
        wr = cv2.VideoWriter(clip+"_rgb.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        for f in rgb_frames: wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
    if depth_color_frames:
        h, w = depth_color_frames[0].shape[:2]
        wr = cv2.VideoWriter(clip+"_depth_color.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w,h))
        for f in depth_color_frames: wr.write(f)
        wr.release()
    if depth_raw_frames:
        np.save(clip+"_depth_raw.npy", np.stack(depth_raw_frames, axis=0))

# ─────────────────────────────────────────────────────────────────────────────
# Worker: pulls frames from camera OR video file, runs YOLO, emits processed frames
# ─────────────────────────────────────────────────────────────────────────────
class FrameWorker(QObject):
    # emits (rgb_bgr annotated, depth_bgr, detection_text)
    frame_ready   = Signal(np.ndarray, np.ndarray, str)
    log_message   = Signal(str)
    finished      = Signal()

    def __init__(self, model_path, source="camera"):
        super().__init__()
        self.model_path = model_path
        self.source     = source   # "camera" or a file path string
        self._running   = False

        # recording state
        self._rec_lock        = threading.Lock()
        self._is_recording    = False
        self._record_until    = 0.0
        self._rgb_buf         = []
        self._depth_color_buf = []
        self._depth_raw_buf   = []

        self._frame_id = 0
        self._t0 = time.time()

        # Persistent 3D identity tracker (track3d_core.py) — (re)created in
        # run() once the model is loaded, so its object vocabulary always
        # matches model.names. See track3d_core.py's module docstring for
        # why identity is keyed by class name rather than YOLO's track id.
        self.tracker = None

        # Set only while replaying a saved clip that has a matching
        # *_depth_raw.npy — see run()'s video-file branch.
        self._clip_csv_writer = None
        self._clip_csv_file   = None
        self._clip_stem       = None

    def start_recording(self):
        with self._rec_lock:
            if self._is_recording:
                return False
            self._is_recording  = True
            self._record_until  = time.time() + CLIP_DURATION_SEC
            self._rgb_buf, self._depth_color_buf, self._depth_raw_buf = [], [], []
            return True

    def stop(self):
        self._running = False

    def _maybe_record(self, rgb, depth_color, depth_mm):
        with self._rec_lock:
            if not self._is_recording:
                return False           # not recording
            if time.time() < self._record_until:
                if rgb         is not None: self._rgb_buf.append(rgb.copy())
                if depth_color is not None: self._depth_color_buf.append(depth_color.copy())
                if depth_mm    is not None: self._depth_raw_buf.append(depth_mm.copy())
                return True            # still recording
            else:
                self._is_recording = False
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                threading.Thread(target=save_clip,
                    args=(list(self._rgb_buf), list(self._depth_color_buf),
                          list(self._depth_raw_buf), ts), daemon=True).start()
                self.log_message.emit(f"Clip saved → clips/{ts}_*.mp4")
                return False           # just finished

    def run(self):
        self._running = True
        self._t0 = time.time()
        model = YOLO(self.model_path)

        # Persistent 3D identity tracker: one slot per class the loaded
        # model knows about (track3d_core.py). Fresh per run() call, so
        # every new stream/clip starts with a clean slate.
        self.tracker = core.PersistentObjectTracker(model.names)

        # ── source: live camera ────────────────────────────────────────────
        if self.source == "camera":
            if not post_encode_config(frame_config_encode(1,1,255,0,2,7,1,0,0)):
                self.log_message.emit("[Error] Could not configure camera.")
                self.finished.emit()
                return
            self.log_message.emit("Camera connected.")

            while self._running:
                raw = get_frame_from_http()
                if raw is None:
                    time.sleep(0.05)
                    continue
                depth_mm, rgb = decode_raw_frame(raw)
                if rgb is None:
                    continue
                self._process_and_emit(model, rgb, depth_mm)

        # ── source: video file ─────────────────────────────────────────────
        else:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                self.log_message.emit(f"[Error] Cannot open: {self.source}")
                self.finished.emit()
                return
            self.log_message.emit(f"Playing: {Path(self.source).name}")

            # A saved clip's RGB video has a matching *_depth_raw.npy right
            # next to it (see save_clip() below). If it's there, load it so
            # Review-tab playback gets real 3D positions instead of always
            # dropping depth for pre-recorded video.
            depth_stack = None
            clip_csv_path = None
            src_path = Path(self.source)
            if src_path.name.endswith("_rgb.mp4"):
                stem = src_path.name[: -len("_rgb.mp4")]
                depth_path = src_path.with_name(f"{stem}_depth_raw.npy")
                if depth_path.exists():
                    try:
                        depth_stack = np.load(depth_path)
                        self.log_message.emit(
                            f"Depth loaded: {depth_path.name} ({depth_stack.shape[0]} frames)")
                    except Exception as e:
                        self.log_message.emit(f"[Warn] Could not load depth ({depth_path.name}): {e}")
                    self._clip_stem = stem
                    clip_csv_path = src_path.with_name(f"{stem}_tracks_3d.csv")
                    self._clip_csv_file = open(clip_csv_path, "w", newline="")
                    self._clip_csv_writer = csv.writer(self._clip_csv_file)
                    self._clip_csv_writer.writerow(core.CSV_HEADER)
                else:
                    self.log_message.emit(
                        f"[Info] No matching {depth_path.name} — playing RGB-only (no 3D).")

            frame_idx = 0
            while self._running:
                ok, bgr = cap.read()
                if not ok:
                    self.log_message.emit("End of video.")
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                depth_mm = None
                if depth_stack is not None and frame_idx < depth_stack.shape[0]:
                    depth_mm = depth_stack[frame_idx]
                self._process_and_emit(model, rgb, depth_mm)
                frame_idx += 1
                time.sleep(1/30)

            cap.release()
            if self._clip_csv_file is not None:
                self._clip_csv_file.close()
                self.log_message.emit(f"3D tracks written → {clip_csv_path.name}")
                self._clip_csv_writer = None
                self._clip_csv_file = None
                self._clip_stem = None

        self.finished.emit()

    def _process_and_emit(self, model, rgb, depth_mm):
        self._frame_id += 1
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_color = depth_to_colormap_bgr(depth_mm) if depth_mm is not None else \
                      np.zeros((DEPTH_H, DEPTH_W, 3), np.uint8)
        depth_color_disp = cv2.resize(depth_color, (RGB_W, RGB_H)) \
                            if depth_mm is not None else np.zeros((RGB_H, RGB_W, 3), np.uint8)

        results = model.track(bgr, persist=True, verbose=False)
        detection_lines = []
        detections = []

        if results and results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids   = results[0].boxes.id.cpu().numpy()
            clss  = results[0].boxes.cls.cpu().numpy()
            confs = (results[0].boxes.conf.cpu().numpy()
                     if results[0].boxes.conf is not None else np.ones(len(boxes)))

            for box, obj_id, cls, cf in zip(boxes, ids, clss, confs):
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                cname  = model.names[int(cls)]
                xyz = None
                if depth_mm is not None:
                    xyz, _dcx, _dcy = core.backproject(cx, cy, depth_mm)
                detections.append(core.Detection(
                    class_name=cname, conf=float(cf), cx=float(cx), cy=float(cy),
                    box=(float(x1), float(y1), float(x2), float(y2)),
                    xyz=xyz, yolo_track_id=int(obj_id),
                ))

        # Persistent identity: keyed by class name, survives an object
        # leaving and re-entering the frame (see track3d_core.py). YOLO's
        # own track id is still recorded per-detection above for reference,
        # but never decides identity — only the tracker's state does.
        timestamp_s = time.time() - self._t0
        objects = self.tracker.update(self._frame_id, timestamp_s, detections) \
                  if self.tracker is not None else {}

        for det in detections:
            x1, y1, x2, y2 = det.box
            colour = (0, 200, 80)
            cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

            obj = objects.get(det.class_name)
            # Prefer the tracker's smoothed, persistent position — it is
            # what survives brief occlusion/reappearance — falling back to
            # this single frame's own reading if the tracker has nothing yet.
            xyz    = obj.xyz if obj is not None and obj.xyz is not None else det.xyz
            status = obj.status if obj is not None else (
                "visible" if det.xyz is not None else "visible_no_depth")

            if xyz is not None:
                X, Y, Z = xyz
                label = f"{det.class_name} [{status}] | X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f}mm"
                cv2.putText(bgr, label, (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
                if depth_mm is not None:
                    # crosshair on depth map — det.cx/cy are already in
                    # RGB-pixel space, which lines up 1:1 with the depth
                    # display once it's resized up to RGB_W×RGB_H.
                    dx_disp = int(np.clip(det.cx, 0, RGB_W - 1))
                    dy_disp = int(np.clip(det.cy, 0, RGB_H - 1))
                    cv2.drawMarker(depth_color_disp, (dx_disp, dy_disp),
                                   (255, 255, 255), cv2.MARKER_CROSS, 14, 2)
                detection_lines.append(
                    f"{det.class_name:<12} [{status:<14}] X:{X:+6.0f} Y:{Y:+6.0f} Z:{Z:6.0f} mm  "
                    f"(yolo id {det.yolo_track_id})")
                log_detection_csv(self._frame_id, det.yolo_track_id, det.class_name,
                                   det.cx, det.cy, X, Y, Z)
            else:
                cv2.putText(bgr, f"{det.class_name} [{status}]", (int(x1), int(y1) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, colour, 1, cv2.LINE_AA)
                detection_lines.append(
                    f"{det.class_name:<12} [{status}]  (yolo id {det.yolo_track_id})  Z: no depth")

        # While replaying a clip with matching depth, also keep a full
        # per-object-per-frame 3D log (same schema as track3d_offline.py's
        # batch output) alongside the live rolling detections_3d.csv above.
        if self._clip_csv_writer is not None:
            for row in self.tracker.frame_rows(self._frame_id, timestamp_s, clip=self._clip_stem):
                self._clip_csv_writer.writerow(row)

        self._maybe_record(rgb, depth_color, depth_mm)
        det_text = "\n".join(detection_lines) if detection_lines else "No detections"
        self.frame_ready.emit(bgr, depth_color_disp, det_text)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: numpy BGR → QPixmap
# ─────────────────────────────────────────────────────────────────────────────
def bgr_to_pixmap(bgr: np.ndarray) -> QPixmap:
    h, w, ch = bgr.shape
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ─────────────────────────────────────────────────────────────────────────────
# Main GUI
# ─────────────────────────────────────────────────────────────────────────────
class TrackerGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Depth + YOLO 3D Tracker")
        self.resize(1400, 820)

        self._worker  = None
        self._thread  = None
        self._model_path = "yolo11n.pt"

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        self.tabs = QTabWidget()
        self.tab_live   = QWidget()
        self.tab_review = QWidget()
        self.tabs.addTab(self.tab_live,   "Live Feed")
        self.tabs.addTab(self.tab_review, "Review")
        root.addWidget(self.tabs)

        init_csv()
        self._build_live_tab()
        self._build_review_tab()

    # ══════════════════════════════════════════════════════════════════════════
    # LIVE TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_live_tab(self):
        layout = QGridLayout(self.tab_live)
        for c in range(12): layout.setColumnStretch(c, 1)
        for r in range(3):  layout.setRowStretch(r, 1)

        # ── video panels (cols 0-7, rows 0-1) ─────────────────────────────
        video_frame = QFrame()
        video_frame.setStyleSheet("QFrame{background:#1a1a1a; border-radius:6px;}")
        vfl = QVBoxLayout(video_frame)

        panel_row = QWidget()
        panel_hl  = QHBoxLayout(panel_row)
        panel_hl.setSpacing(4)

        # RGB panel
        rgb_wrap = QFrame()
        rgb_wrap.setStyleSheet("QFrame{background:#111;border-radius:4px;}")
        rl = QVBoxLayout(rgb_wrap)
        rgb_hdr = QLabel("RGB + YOLO")
        rgb_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgb_hdr.setStyleSheet("color:#ccc; font-weight:bold;")
        self.rgb_label = QLabel()
        self.rgb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rgb_label.setMinimumSize(560, 360)
        self.rgb_label.setStyleSheet("background:#000;")
        rl.addWidget(rgb_hdr)
        rl.addWidget(self.rgb_label, 1)

        # Depth panel
        depth_wrap = QFrame()
        depth_wrap.setStyleSheet("QFrame{background:#111;border-radius:4px;}")
        dl = QVBoxLayout(depth_wrap)
        depth_hdr = QLabel("Depth Map (jet_r)")
        depth_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        depth_hdr.setStyleSheet("color:#ccc; font-weight:bold;")
        self.depth_label = QLabel()
        self.depth_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.depth_label.setMinimumSize(560, 360)
        self.depth_label.setStyleSheet("background:#000;")
        dl.addWidget(depth_hdr)
        dl.addWidget(self.depth_label, 1)

        panel_hl.addWidget(rgb_wrap,   1)
        panel_hl.addWidget(depth_wrap, 1)
        vfl.addWidget(panel_row, 1)
        layout.addWidget(video_frame, 0, 0, 2, 8)

        # ── controls panel (cols 8-11, rows 0-1) ──────────────────────────
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        cfl = QVBoxLayout(ctrl_frame)
        cfl.setSpacing(8)
        cfl.setContentsMargins(10, 10, 10, 10)

        # ─ Source selection ──────────────────────────────────────────────
        src_hdr = QLabel("Source")
        src_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Live Camera", "Video File…"])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        self._video_path = None

        self.source_path_label = QLabel("")
        self.source_path_label.setStyleSheet("color:#888; font-size:10px;")
        self.source_path_label.setWordWrap(True)

        # ─ Model selection ───────────────────────────────────────────────
        model_hdr = QLabel("YOLO Model")
        model_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        self.model_combo = QComboBox()
        self._populate_model_combo()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)

        browse_model_btn = QPushButton("Browse model…")
        browse_model_btn.clicked.connect(self._browse_model)

        # ─ Stream controls ───────────────────────────────────────────────
        stream_hdr = QLabel("Stream")
        stream_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        self.start_btn = QPushButton("▶  Start")
        self.stop_btn  = QPushButton("■  Stop")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self._start_stream)
        self.stop_btn.clicked.connect(self._stop_stream)
        self.start_btn.setStyleSheet("background:#1e7e34; color:white; padding:6px; border-radius:4px;")
        self.stop_btn.setStyleSheet("background:#7e1e1e; color:white; padding:6px; border-radius:4px;")

        # ─ Recording controls ────────────────────────────────────────────
        rec_hdr = QLabel("Recording")
        rec_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        self.rec_btn = QPushButton(f"⏺  Record {CLIP_DURATION_SEC}s Clip")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self._start_recording)
        self.rec_btn.setStyleSheet("background:#444; color:white; padding:6px; border-radius:4px;")

        self.rec_status = QLabel("Not recording")
        self.rec_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rec_status.setStyleSheet("color:#888; font-size:10px;")

        # ─ Status ────────────────────────────────────────────────────────
        self.stream_status = QLabel("● Stopped")
        self.stream_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.stream_status.setStyleSheet("color:#e55; font-weight:bold;")

        cfl.addWidget(src_hdr)
        cfl.addWidget(self.source_combo)
        cfl.addWidget(self.source_path_label)
        cfl.addSpacing(6)
        cfl.addWidget(model_hdr)
        cfl.addWidget(self.model_combo)
        cfl.addWidget(browse_model_btn)
        cfl.addSpacing(6)
        cfl.addWidget(stream_hdr)
        cfl.addWidget(self.start_btn)
        cfl.addWidget(self.stop_btn)
        cfl.addWidget(self.stream_status)
        cfl.addSpacing(6)
        cfl.addWidget(rec_hdr)
        cfl.addWidget(self.rec_btn)
        cfl.addWidget(self.rec_status)
        cfl.addStretch(1)

        layout.addWidget(ctrl_frame, 0, 8, 2, 4)

        # ── bottom row: detections log + terminal ──────────────────────
        det_frame = QFrame()
        det_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        dfl = QVBoxLayout(det_frame)
        dfl.setContentsMargins(8,8,8,8)
        det_hdr = QLabel("Live Detections (3D)")
        det_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        det_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.det_output = QTextEdit()
        self.det_output.setReadOnly(True)
        self.det_output.setFont(QFont("Courier New", 9))
        self.det_output.setStyleSheet("background:#1a1a1a; color:#7fc97f;")
        dfl.addWidget(det_hdr)
        dfl.addWidget(self.det_output, 1)
        layout.addWidget(det_frame, 2, 0, 1, 6)

        term_frame = QFrame()
        term_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        tfl = QVBoxLayout(term_frame)
        tfl.setContentsMargins(8,8,8,8)
        term_hdr = QLabel("Terminal")
        term_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        term_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.terminal = QTextEdit()
        self.terminal.setReadOnly(True)
        self.terminal.setFont(QFont("Courier New", 9))
        self.terminal.setStyleSheet("background:#1a1a1a; color:#ccc;")
        tfl.addWidget(term_hdr)
        tfl.addWidget(self.terminal, 1)
        layout.addWidget(term_frame, 2, 6, 1, 6)

    # ══════════════════════════════════════════════════════════════════════════
    # REVIEW TAB
    # ══════════════════════════════════════════════════════════════════════════
    def _build_review_tab(self):
        layout = QGridLayout(self.tab_review)
        for c in range(12): layout.setColumnStretch(c, 1)
        for r in range(3):  layout.setRowStretch(r, 1)

        # ── saved clips browser (cols 0-7, row 0) ─────────────────────────
        clips_frame = QFrame()
        clips_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        cfl = QVBoxLayout(clips_frame)
        cfl.setContentsMargins(10,10,10,10)

        clips_hdr = QLabel("Saved Clips")
        clips_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        clips_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.clips_list = QListWidget()
        self.clips_list.setStyleSheet("background:#1a1a1a; color:#ccc;")
        self._refresh_clips()

        clips_toolbar = QToolBar()
        refresh_btn = QToolButton(); refresh_btn.setText("Refresh")
        refresh_btn.clicked.connect(self._refresh_clips)
        play_btn    = QToolButton(); play_btn.setText("▶ Play in Live tab")
        play_btn.clicked.connect(self._play_selected_clip)
        clips_toolbar.addWidget(refresh_btn)
        clips_toolbar.addWidget(play_btn)

        cfl.addWidget(clips_hdr)
        cfl.addWidget(self.clips_list, 1)
        cfl.addWidget(clips_toolbar)
        layout.addWidget(clips_frame, 0, 0, 2, 8)

        # ── CSV log viewer (cols 8-11, rows 0-1) ──────────────────────────
        csv_frame = QFrame()
        csv_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        cvfl = QVBoxLayout(csv_frame)
        cvfl.setContentsMargins(10,10,10,10)

        csv_hdr = QLabel("Detection Log (CSV)")
        csv_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        csv_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.csv_output = QTextEdit()
        self.csv_output.setReadOnly(True)
        self.csv_output.setFont(QFont("Courier New", 8))
        self.csv_output.setStyleSheet("background:#1a1a1a; color:#7fc97f;")

        csv_toolbar = QToolBar()
        csv_refresh = QToolButton(); csv_refresh.setText("Refresh Log")
        csv_refresh.clicked.connect(self._refresh_csv)
        csv_clear   = QToolButton(); csv_clear.setText("Clear Log")
        csv_clear.clicked.connect(self._clear_csv)
        csv_toolbar.addWidget(csv_refresh)
        csv_toolbar.addWidget(csv_clear)

        cvfl.addWidget(csv_hdr)
        cvfl.addWidget(self.csv_output, 1)
        cvfl.addWidget(csv_toolbar)
        layout.addWidget(csv_frame, 0, 8, 2, 4)

        # ── review terminal (row 2, full width) ──────────────────────────
        rterm_frame = QFrame()
        rterm_frame.setStyleSheet("QFrame{background:#2b2b2b; border-radius:6px;}")
        rtfl = QVBoxLayout(rterm_frame)
        rtfl.setContentsMargins(8,8,8,8)
        rterm_hdr = QLabel("Review Terminal")
        rterm_hdr.setStyleSheet("color:#aaa; font-weight:bold;")
        rterm_hdr.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.review_terminal = QTextEdit()
        self.review_terminal.setReadOnly(True)
        self.review_terminal.setFont(QFont("Courier New", 9))
        self.review_terminal.setStyleSheet("background:#1a1a1a; color:#ccc;")
        rtfl.addWidget(rterm_hdr)
        rtfl.addWidget(self.review_terminal, 1)
        layout.addWidget(rterm_frame, 2, 0, 1, 12)

    # ══════════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════════
    def _populate_model_combo(self):
        self.model_combo.clear()
        # add any .pt files in cwd
        for f in Path(".").glob("*.pt"):
            self.model_combo.addItem(str(f))
        if self.model_combo.count() == 0:
            self.model_combo.addItem("yolo11n.pt")
        self._model_path = self.model_combo.currentText()

    def _on_model_changed(self, text):
        self._model_path = text
        self.log(f"Model set → {text}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO weights", "", "Weights (*.pt)")
        if path:
            self._model_path = path
            if self.model_combo.findText(path) == -1:
                self.model_combo.addItem(path)
            self.model_combo.setCurrentText(path)

    def _on_source_changed(self, idx):
        if idx == 1:   # "Video File…"
            path, _ = QFileDialog.getOpenFileName(self, "Select video file", "",
                                                  "Video (*.mp4 *.avi *.mov *.mkv)")
            if path:
                self._video_path = path
                self.source_path_label.setText(Path(path).name)
            else:
                self.source_combo.setCurrentIndex(0)
        else:
            self._video_path = None
            self.source_path_label.setText("")

    def _start_stream(self):
        if self._thread and self._thread.isRunning():
            return

        source = "camera" if self.source_combo.currentIndex() == 0 \
                 else (self._video_path or "camera")

        self._worker = FrameWorker(self._model_path, source)
        self._thread = QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.frame_ready.connect(self._on_frame)
        self._worker.log_message.connect(self.log)
        self._worker.finished.connect(self._on_worker_done)
        self._thread.start()

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.rec_btn.setEnabled(True)
        self.stream_status.setText("● Streaming")
        self.stream_status.setStyleSheet("color:#3c3; font-weight:bold;")
        self.log(f"Stream started | source={source} | model={self._model_path}")

    def _stop_stream(self):
        if self._worker:
            self._worker.stop()
        self.log("Stream stopped.")

    def _on_worker_done(self):
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self.stream_status.setText("● Stopped")
        self.stream_status.setStyleSheet("color:#e55; font-weight:bold;")

    def _start_recording(self):
        if self._worker:
            started = self._worker.start_recording()
            if started:
                self.rec_status.setText(f"Recording {CLIP_DURATION_SEC}s…")
                self.rec_status.setStyleSheet("color:#e55; font-size:10px; font-weight:bold;")
                self.log(f"Recording {CLIP_DURATION_SEC}s clip…")
                QTimer.singleShot(CLIP_DURATION_SEC*1000+500, self._rec_done)
            else:
                self.log("Already recording.")

    def _rec_done(self):
        self.rec_status.setText("Clip saved ✓")
        self.rec_status.setStyleSheet("color:#3c3; font-size:10px;")
        self._refresh_clips()

    def _on_frame(self, bgr: np.ndarray, depth_bgr: np.ndarray, det_text: str):
        # Display RGB
        px = bgr_to_pixmap(bgr)
        self.rgb_label.setPixmap(
            px.scaled(self.rgb_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation))
        # Display Depth
        px2 = bgr_to_pixmap(depth_bgr)
        self.depth_label.setPixmap(
            px2.scaled(self.depth_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                       Qt.TransformationMode.SmoothTransformation))
        # Detection log (replace contents each frame)
        self.det_output.setPlainText(det_text)

    def _refresh_clips(self):
        self.clips_list.clear()
        # Only list the RGB clip itself (not its *_depth_color.mp4 preview) —
        # that's the file _play_selected_clip / FrameWorker actually open,
        # and it's what has a matching *_depth_raw.npy for 3D tracking.
        for f in sorted(Path(OUTPUT_DIR).glob("*_rgb.mp4")):
            depth_path = f.with_name(f.name[: -len("_rgb.mp4")] + "_depth_raw.npy")
            item = QListWidgetItem(str(f))
            item.setToolTip("3D tracking available (matching depth file found)"
                             if depth_path.exists() else
                             "RGB only — no matching *_depth_raw.npy")
            self.clips_list.addItem(item)

    def _play_selected_clip(self):
        items = self.clips_list.selectedItems()
        if not items:
            return
        path = items[0].text()
        self._video_path = path
        self.source_combo.setCurrentIndex(1)
        self.source_path_label.setText(Path(path).name)
        self.tabs.setCurrentWidget(self.tab_live)
        self._stop_stream()
        QTimer.singleShot(300, self._start_stream)
        self.review_log(f"Playing clip: {path}")

    def _refresh_csv(self):
        if not os.path.exists(CSV_PATH):
            self.csv_output.setPlainText("No detections logged yet.")
            return
        with open(CSV_PATH) as f:
            lines = f.readlines()
        # show last 200 lines
        self.csv_output.setPlainText("".join(lines[-200:]))

    def _clear_csv(self):
        init_csv()   # re-creates header only
        self.csv_output.clear()
        self.review_log("Detection log cleared.")

    def log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.terminal.append(f"> {ts}  {text}")

    def review_log(self, text):
        ts = datetime.now().strftime("%H:%M:%S")
        self.review_terminal.append(f"> {ts}  {text}")

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
        if self._thread:
            self._thread.quit()
            self._thread.wait()
        event.accept()


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark palette matching the E23 GUI aesthetic
    palette = app.palette()
    palette.setColor(palette.ColorRole.Window,          QColor(45,  45,  45))
    palette.setColor(palette.ColorRole.WindowText,      QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Base,            QColor(30,  30,  30))
    palette.setColor(palette.ColorRole.AlternateBase,   QColor(50,  50,  50))
    palette.setColor(palette.ColorRole.ToolTipBase,     QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.ToolTipText,     QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Text,            QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Button,          QColor(60,  60,  60))
    palette.setColor(palette.ColorRole.ButtonText,      QColor(220, 220, 220))
    palette.setColor(palette.ColorRole.Highlight,       QColor(42,  130, 218))
    palette.setColor(palette.ColorRole.HighlightedText, QColor(0,   0,   0))
    app.setPalette(palette)

    win = TrackerGUI()
    win.show()
    sys.exit(app.exec())