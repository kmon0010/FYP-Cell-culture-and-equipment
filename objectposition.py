"""
Depth-fused YOLO 3D Object Tracker
------------------------------------
Pulls live RGB + depth frames from the camera (camera_with_recording.py),
runs YOLO tracking on the RGB stream, back-projects each detection's bounding-box
centre through the depth map to a real-world (X, Y, Z) coordinate using camera
intrinsics, overlays the result on the display, and logs every detection to CSV.

Controls (OpenCV window):
  q  — quit
  c  — clear selected object
  r  — start/stop a fixed-duration clip recording (RGB + depth colour MP4 + raw .npy)

Camera intrinsics
-----------------
Edit the INTRINSICS block below with your values before running.
"""

import requests
import struct
import threading
import time
import os
import csv
from datetime import datetime

import numpy as np
import cv2
import matplotlib.cm as cm
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# ──────────────────────────────────────────────
# Camera connection
# ──────────────────────────────────────────────
HOST = "192.168.233.1"
PORT = 80

# ──────────────────────────────────────────────
# Camera intrinsics  ← fill these in
# ──────────────────────────────────────────────
# Depth image is 320×240; intrinsics should match that resolution.
FX = 200.0   # focal length x (pixels)
FY = 200.0   # focal length y (pixels)
CX = 160.0   # principal point x
CY = 120.0   # principal point y

# ──────────────────────────────────────────────
# Depth remapping  (raw sensor units → mm)
# ──────────────────────────────────────────────
DEPTH_OLD_MIN, DEPTH_OLD_MAX = 16, 255
DEPTH_NEW_MIN, DEPTH_NEW_MAX = 150, 1500   # mm

# ──────────────────────────────────────────────
# Recording settings
# ──────────────────────────────────────────────
CLIP_DURATION_SEC = 5
ESTIMATED_FPS     = 10
OUTPUT_DIR        = "clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# CSV logging
# ──────────────────────────────────────────────
CSV_PATH = os.path.join(OUTPUT_DIR, "detections_3d.csv")
_csv_lock = threading.Lock()

def _init_csv():
    if not os.path.exists(CSV_PATH):
        with open(CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "frame_id", "track_id", "class",
                             "cx_px", "cy_px",        # bbox centre in RGB pixels
                             "depth_cx_px", "depth_cy_px",  # corresponding depth pixel
                             "X_mm", "Y_mm", "Z_mm"])

def log_detection(frame_id, track_id, class_name,
                  cx_px, cy_px, dcx, dcy, X, Y, Z):
    ts = datetime.now().isoformat(timespec="milliseconds")
    with _csv_lock:
        with open(CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([ts, frame_id, track_id, class_name,
                             round(cx_px, 1), round(cy_px, 1),
                             round(dcx, 1), round(dcy, 1),
                             round(X, 1), round(Y, 1), round(Z, 1)])

# ──────────────────────────────────────────────
# Camera protocol helpers  (from camera_with_recording.py)
# ──────────────────────────────────────────────
def frame_config_decode(raw):
    return struct.unpack("<BBBBBBBBi", raw)

def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255,
                         ir_mode=1, status_mode=2, status_mask=7,
                         rgb_mode=1, rgb_res=0, expose_time=0):
    return struct.pack("<BBBBBBBBi",
                       trigger_mode, deep_mode, deep_shift, ir_mode,
                       status_mode, status_mask, rgb_mode, rgb_res, expose_time)

def frame_payload_decode(frame_data: bytes, cfg: tuple):
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    payload = frame_data[8:]

    deepth_size = (320 * 240 * 2) >> cfg[1]
    depth_bytes = payload[:deepth_size] if deepth_size else None
    payload = payload[deepth_size:]

    ir_size = (320 * 240 * 2) >> cfg[3]
    payload = payload[ir_size:]   # skip IR

    status_size = (320 * 240 // 8) * (
        16 if cfg[4] == 0 else 2 if cfg[4] == 1 else 8 if cfg[4] == 2 else 1)
    payload = payload[status_size:]   # skip status

    rgb_size = len(payload)
    rgb_bytes = payload[:rgb_size] if rgb_size else None

    if rgb_bytes and cfg[6] == 1:
        jpeg = cv2.imdecode(np.frombuffer(rgb_bytes, "uint8", rgb_size),
                            cv2.IMREAD_COLOR)
        rgb_bytes = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB).tobytes() if jpeg is not None else None

    return depth_bytes, rgb_bytes

def post_encode_config(config=frame_config_encode(), host=HOST, port=PORT):
    r = requests.post(f"http://{host}:{port}/set_cfg", config)
    return r.status_code == requests.codes.ok

def get_frame_from_http(host=HOST, port=PORT):
    r = requests.get(f"http://{host}:{port}/getdeep")
    if r.status_code == requests.codes.ok:
        return r.content
    return None

def decode_raw_frame(raw: bytes):
    """Return (depth_mm float32 240×320, rgb uint8 480×640×3) or Nones."""
    cfg = frame_config_decode(raw[16:28])
    depth_bytes, rgb_bytes = frame_payload_decode(raw[28:], cfg)

    depth_mm = None
    if depth_bytes:
        dtype = "uint16" if cfg[1] == 0 else "uint8"
        raw_d = np.frombuffer(depth_bytes, dtype).reshape(240, 320).astype(float)
        depth_mm = ((raw_d - DEPTH_OLD_MIN) * (DEPTH_NEW_MAX - DEPTH_NEW_MIN) /
                    (DEPTH_OLD_MAX - DEPTH_OLD_MIN) + DEPTH_NEW_MIN).astype(np.float32)

    rgb = None
    if rgb_bytes:
        rgb = np.frombuffer(rgb_bytes, "uint8").reshape(480, 640, 3)

    return depth_mm, rgb

# ──────────────────────────────────────────────
# 3D back-projection
# ──────────────────────────────────────────────
# The depth map is 320×240 while RGB is 640×480 (2× scale).
# We scale the RGB bbox centre down to depth-image coordinates before sampling.
RGB_W, RGB_H     = 640, 480
DEPTH_W, DEPTH_H = 320, 240
SCALE_X = DEPTH_W / RGB_W   # 0.5
SCALE_Y = DEPTH_H / RGB_H   # 0.5

def backproject(cx_rgb, cy_rgb, depth_mm: np.ndarray):
    """
    Given a bounding-box centre in RGB pixel coords and the depth map (mm),
    return (X_mm, Y_mm, Z_mm) in camera space, plus the depth-image pixel used.
    Returns None if depth is zero/invalid.
    """
    dcx = int(np.clip(cx_rgb * SCALE_X, 0, DEPTH_W - 1))
    dcy = int(np.clip(cy_rgb * SCALE_Y, 0, DEPTH_H - 1))
    Z = float(depth_mm[dcy, dcx])
    if Z <= 0:
        return None, dcx, dcy
    X = (dcx - CX) * Z / FX
    Y = (dcy - CY) * Z / FY
    return (X, Y, Z), dcx, dcy

# ──────────────────────────────────────────────
# Depth colourmap helper
# ──────────────────────────────────────────────
def depth_to_colormap_bgr(depth_mm: np.ndarray) -> np.ndarray:
    norm = np.clip((depth_mm - DEPTH_NEW_MIN) / (DEPTH_NEW_MAX - DEPTH_NEW_MIN), 0, 1)
    colored = (cm.jet_r(norm)[:, :, :3] * 255).astype(np.uint8)
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)

# ──────────────────────────────────────────────
# Clip recording (background thread)
# ──────────────────────────────────────────────
_rec_lock         = threading.Lock()
_is_recording     = False
_record_until     = 0.0
_rgb_buf          = []
_depth_color_buf  = []
_depth_raw_buf    = []

def _save_clip(rgb_frames, depth_color_frames, depth_raw_frames, ts):
    clip = os.path.join(OUTPUT_DIR, ts)
    fps  = ESTIMATED_FPS

    if rgb_frames:
        h, w = rgb_frames[0].shape[:2]
        wr = cv2.VideoWriter(clip + "_rgb.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in rgb_frames:
            wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        wr.release()
        print(f"[Rec] Saved {clip}_rgb.mp4  ({len(rgb_frames)} frames)")

    if depth_color_frames:
        h, w = depth_color_frames[0].shape[:2]
        wr = cv2.VideoWriter(clip + "_depth_color.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in depth_color_frames:
            wr.write(f)
        wr.release()
        print(f"[Rec] Saved {clip}_depth_color.mp4")

    if depth_raw_frames:
        path = clip + "_depth_raw.npy"
        np.save(path, np.stack(depth_raw_frames, axis=0))
        print(f"[Rec] Saved {path}  shape={np.stack(depth_raw_frames).shape}")

def _maybe_record(rgb, depth_color, depth_mm):
    global _is_recording, _record_until
    with _rec_lock:
        if not _is_recording:
            return
        if time.time() < _record_until:
            if rgb is not None:         _rgb_buf.append(rgb.copy())
            if depth_color is not None: _depth_color_buf.append(depth_color.copy())
            if depth_mm is not None:    _depth_raw_buf.append(depth_mm.copy())
        else:
            _is_recording = False
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            threading.Thread(
                target=_save_clip,
                args=(list(_rgb_buf), list(_depth_color_buf), list(_depth_raw_buf), ts),
                daemon=True,
            ).start()
            print("[Rec] Clip done — saving in background. Press 'r' to record again.")

# ──────────────────────────────────────────────
# Main tracker class
# ──────────────────────────────────────────────
class DepthYOLOTracker:
    def __init__(self, model="yolo11n.pt", crop_size=(300, 300)):
        self.model      = YOLO(model)
        self.names      = self.model.names
        self.crop_size  = crop_size
        self.crop_pad   = 5
        self.crop_margin = 5

        self.selected_id  = None
        self.current_data = None   # (boxes_xyxy, ids, depth_mm) for mouse CB
        self.frame_id     = 0

        self.window = "Depth + YOLO 3D Tracker"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, 1280, 540)
        cv2.setMouseCallback(self.window, self._mouse_cb)

    # ── mouse ──────────────────────────────────
    def _mouse_cb(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.current_data is None:
            return
        boxes, ids, _ = self.current_data
        self.selected_id = None
        if boxes is not None and ids is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(ids[i])
                    break

    # ── crop overlay ───────────────────────────
    def _crop_overlay(self, im0, box):
        h, w = im0.shape[:2]
        x1, y1, x2, y2 = box.astype(int)
        crop = im0[max(0, y1 - self.crop_pad):min(h, y2 + self.crop_pad),
                   max(0, x1 - self.crop_pad):min(w, x2 + self.crop_pad)]
        if crop.size == 0:
            return im0
        ch, cw = crop.shape[:2]
        scale  = min(self.crop_size[0] / cw, self.crop_size[1] / ch)
        crop   = cv2.resize(crop, (int(cw * scale), int(ch * scale)))
        m = self.crop_margin
        y_end = m + crop.shape[0]
        x_start = w - crop.shape[1] - m
        if y_end <= h and x_start >= 0:
            cv2.rectangle(im0, (x_start - 2, m - 2),
                          (x_start + crop.shape[1] + 2, y_end + 2), (68, 243, 0), 4)
            im0[m:y_end, x_start:x_start + crop.shape[1]] = crop
        return im0

    # ── draw 3D label ──────────────────────────
    @staticmethod
    def _draw_3d_label(im0, cx, cy, xyz, track_id, class_name, colour):
        X, Y, Z = xyz
        label  = f"ID{track_id} {class_name}"
        coords = f"X:{X:+.0f} Y:{Y:+.0f} Z:{Z:.0f} mm"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 1
        pad        = 4

        (tw, th), _ = cv2.getTextSize(coords, font, font_scale, thickness)
        (lw, lh), _ = cv2.getTextSize(label,  font, font_scale, thickness)
        box_w = max(tw, lw) + pad * 2
        box_h = (th + lh) + pad * 3

        bx1 = int(cx) - box_w // 2
        by1 = int(cy) - box_h - 8
        bx2, by2 = bx1 + box_w, by1 + box_h

        # clamp to frame
        h, w = im0.shape[:2]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)

        overlay = im0.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, im0, 0.45, 0, im0)
        cv2.rectangle(im0, (bx1, by1), (bx2, by2), colour, 1)
        cv2.putText(im0, label,  (bx1 + pad, by1 + pad + lh),
                    font, font_scale, colour,    thickness, cv2.LINE_AA)
        cv2.putText(im0, coords, (bx1 + pad, by1 + pad + lh + th + pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # ── status bar ─────────────────────────────
    @staticmethod
    def _draw_status(im0, recording):
        h, w = im0.shape[:2]
        bar = np.zeros((28, w, 3), dtype=np.uint8)
        rec_txt = "  [REC]" if recording else ""
        hint = f"Click object to select | r=record{rec_txt} | c=clear | q=quit"
        cv2.putText(bar, hint, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (200, 200, 200), 1, cv2.LINE_AA)
        if recording:
            cv2.putText(bar, "[REC]", (w - 70, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        return np.vstack([im0, bar])

    # ── main loop ──────────────────────────────
    def run(self):
        global _is_recording, _record_until, _rgb_buf, _depth_color_buf, _depth_raw_buf

        _init_csv()
        print(f"[Info] Logging detections to: {CSV_PATH}")
        print(f"[Info] Press 'r' to record a {CLIP_DURATION_SEC}s clip.")

        while True:
            raw = get_frame_from_http()
            if raw is None:
                continue

            depth_mm, rgb = decode_raw_frame(raw)
            if rgb is None:
                continue

            self.frame_id += 1
            depth_color_bgr = depth_to_colormap_bgr(depth_mm) if depth_mm is not None else None

            # ── YOLO tracking on BGR copy ──────
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            results = self.model.track(bgr, persist=True, verbose=False)

            ann = Annotator(bgr, line_width=2)
            display_depth = cv2.resize(depth_color_bgr, (RGB_W, RGB_H)) \
                            if depth_color_bgr is not None else np.zeros((RGB_H, RGB_W, 3), np.uint8)

            if results and results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                ids   = results[0].boxes.id.cpu().numpy()
                clss  = results[0].boxes.cls.cpu().numpy()

                self.current_data = (boxes, ids, depth_mm)

                for box, obj_id, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx_rgb = (x1 + x2) / 2
                    cy_rgb = (y1 + y2) / 2
                    tid    = int(obj_id)
                    cname  = self.names[int(cls)]
                    col    = colors(int(cls), True)

                    # Draw standard YOLO box
                    ann.box_label(box, label=f"ID{tid} {cname}", color=col)

                    # 3D position
                    xyz, dcx, dcy = backproject(cx_rgb, cy_rgb, depth_mm) \
                                    if depth_mm is not None else (None, 0, 0)

                    if xyz is not None:
                        self._draw_3d_label(bgr, cx_rgb, cy_rgb, xyz, tid, cname, col)
                        # mark depth sample point on depth display
                        dcx_disp = int(dcx / SCALE_X)
                        dcy_disp = int(dcy / SCALE_Y)
                        cv2.drawMarker(display_depth, (dcx_disp, dcy_disp),
                                       (255, 255, 255), cv2.MARKER_CROSS, 12, 2)
                        log_detection(self.frame_id, tid, cname,
                                      cx_rgb, cy_rgb, dcx, dcy, *xyz)

                # Crop overlay for selected object
                if self.selected_id is not None:
                    for i, oid in enumerate(ids):
                        if int(oid) == self.selected_id:
                            bgr = self._crop_overlay(bgr, boxes[i])
                            break
                    else:
                        self.selected_id = None

            # ── side-by-side display ───────────
            combined = np.hstack([bgr, display_depth])
            combined = self._draw_status(combined, _is_recording)

            # ── recording ──────────────────────
            _maybe_record(rgb, depth_color_bgr, depth_mm)

            cv2.imshow(self.window, combined)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("c"):
                self.selected_id = None
                print("[Info] Selection cleared.")
            elif key == ord("r"):
                with _rec_lock:
                    if _is_recording:
                        print("[Rec] Already recording.")
                    else:
                        _is_recording    = True
                        _record_until    = time.time() + CLIP_DURATION_SEC
                        _rgb_buf         = []
                        _depth_color_buf = []
                        _depth_raw_buf   = []
                        print(f"[Rec] Started — capturing {CLIP_DURATION_SEC}s …")

        cv2.destroyAllWindows()
        print("[Info] Done.")


# ──────────────────────────────────────────────
if __name__ == "__main__":
    if not post_encode_config(frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)):
        print("[Error] Could not configure camera. Check HOST/PORT.")
    else:
        tracker = DepthYOLOTracker(
           #model="weights.pt",   # swap for your weights
           model="yolo11n.pt",
            crop_size=(300, 300),
        )
        tracker.run()