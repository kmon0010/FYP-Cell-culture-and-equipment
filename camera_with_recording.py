from PIL import Image  # imaging library
import requests  # http library
import matplotlib.pyplot as plt
import struct
import numpy as np
import cv2
import threading
import time
import os
from datetime import datetime
import matplotlib.cm as cm


def frame_config_decode(frame_config):
    '''
        @frame_config bytes

        @return fields, tuple (trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)
    '''
    return struct.unpack("<BBBBBBBBi", frame_config)


def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255, ir_mode=1, status_mode=2, status_mask=7, rgb_mode=1, rgb_res=0, expose_time=0):
    return struct.pack("<BBBBBBBBi",
                       trigger_mode, deep_mode, deep_shift, ir_mode, status_mode, status_mask, rgb_mode, rgb_res, expose_time)


def frame_payload_decode(frame_data: bytes, with_config: tuple):
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]
    # 0:16bit 1:8bit, resolution: 320*240
    deepth_size = (320*240*2) >> with_config[1]
    deepth_img = struct.unpack("<%us" % deepth_size, frame_payload[:deepth_size])[
        0] if 0 != deepth_size else None
    frame_payload = frame_payload[deepth_size:]

    # 0:16bit 1:8bit, resolution: 320*240
    ir_size = (320*240*2) >> with_config[3]
    ir_img = struct.unpack("<%us" % ir_size, frame_payload[:ir_size])[
        0] if 0 != ir_size else None
    frame_payload = frame_payload[ir_size:]

    status_size = (320*240//8) * (16 if 0 == with_config[4] else
                                  2 if 1 == with_config[4] else 8 if 2 == with_config[4] else 1)
    status_img = struct.unpack("<%us" % status_size, frame_payload[:status_size])[
        0] if 0 != status_size else None
    frame_payload = frame_payload[status_size:]

    assert(deep_data_size == deepth_size+ir_size+status_size)

    rgb_size = len(frame_payload)
    assert(rgb_data_size == rgb_size)
    rgb_img = struct.unpack("<%us" % rgb_size, frame_payload[:rgb_size])[
        0] if 0 != rgb_size else None

    if (not rgb_img is None) and (1 == with_config[6]):
        jpeg = cv2.imdecode(np.frombuffer(
            rgb_img, 'uint8', rgb_size), cv2.IMREAD_COLOR)
        if not jpeg is None:
            rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
            rgb_img = rgb.tobytes()
        else:
            rgb_img = None

    return (deepth_img, ir_img, status_img, rgb_img)


HOST = '192.168.233.1'
PORT = 80

# --- Recording settings ---
CLIP_DURATION_SEC = 20     # Duration of each clip in seconds
ESTIMATED_FPS = 10        # Approximate capture FPS (used for VideoWriter)
OUTPUT_DIR = "clips"      # Folder to save clips into
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Depth remapping range (must match show_frame)
DEPTH_OLD_MIN = 16
DEPTH_OLD_MAX = 255
DEPTH_NEW_MIN = 150
DEPTH_NEW_MAX = 1500


def post_encode_config(config=frame_config_encode(), host=HOST, port=PORT, timeout=5):
    # A bare requests call with no timeout hangs for the OS-level TCP
    # connect timeout (~20s on Windows) and, uncaught, kills the whole
    # script the moment the camera is briefly unreachable (still booting,
    # a loose USB-C cable, its virtual network adapter not up yet). Fail
    # fast and let the caller decide whether to retry instead.
    try:
        r = requests.post('http://{}:{}/set_cfg'.format(host, port), config, timeout=timeout)
    except requests.exceptions.RequestException as e:
        print(f"[Error] Could not reach camera at {host}:{port} to configure it ({e.__class__.__name__}).")
        return False
    if(r.status_code == requests.codes.ok):
        return True
    return False


def get_frame_from_http(host=HOST, port=PORT, timeout=5):
    try:
        r = requests.get('http://{}:{}/getdeep'.format(host, port), timeout=timeout)
    except requests.exceptions.RequestException:
        return None
    if(r.status_code == requests.codes.ok):
        deepimg = r.content
        (frameid, stamp_msec) = struct.unpack('<QQ', deepimg[0:8+8])
        return deepimg
    return None


def depth_to_colormap(depth_array):
    """Convert a mapped depth array (mm floats) to a uint8 BGR colormap image for VideoWriter."""
    # Normalise to 0-255 for colormap
    norm = np.clip((depth_array - DEPTH_NEW_MIN) / (DEPTH_NEW_MAX - DEPTH_NEW_MIN), 0, 1)
    colored = (cm.jet_r(norm)[:, :, :3] * 255).astype(np.uint8)   # RGB uint8
    return cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)                # BGR for OpenCV


def save_clip(rgb_frames, depth_frames_color, depth_frames_raw, timestamp):
    """Write collected frames to MP4 files and raw depth to .npy — runs in a background thread."""
    clip_name = os.path.join(OUTPUT_DIR, timestamp)
    fps = ESTIMATED_FPS

    # --- RGB MP4 ---
    if rgb_frames:
        h, w = rgb_frames[0].shape[:2]
        rgb_path = clip_name + "_rgb.mp4"
        writer = cv2.VideoWriter(rgb_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in rgb_frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[Recording] Saved RGB clip: {rgb_path}  ({len(rgb_frames)} frames)")

    # --- Depth colorised MP4 ---
    if depth_frames_color:
        h, w = depth_frames_color[0].shape[:2]
        depth_path = clip_name + "_depth_color.mp4"
        writer = cv2.VideoWriter(depth_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in depth_frames_color:
            writer.write(frame)
        writer.release()
        print(f"[Recording] Saved depth (colorised) clip: {depth_path}  ({len(depth_frames_color)} frames)")

    # --- Raw depth .npy ---
    if depth_frames_raw:
        raw_path = clip_name + "_depth_raw.npy"
        np.save(raw_path, np.stack(depth_frames_raw, axis=0))
        print(f"[Recording] Saved raw depth array: {raw_path}  shape={np.stack(depth_frames_raw).shape}")


# Shared state between the main loop and keyboard listener
recording_lock = threading.Lock()
is_recording = False
record_until = 0.0          # time.time() value at which recording stops
rgb_buffer = []
depth_color_buffer = []
depth_raw_buffer = []

# Each recording method below registers itself into these three lists, so
# commenting out a whole method block (from its ">>> METHOD" line to its
# "<<< METHOD" line) removes it completely - no other edits needed.
KEY_HANDLERS = []   # key-press handlers connected to the figure
KEY_HINTS = []      # short "press X to ..." text shown in the RGB title
STARTUP_INFO = []   # lines printed once at start-up


# >>> METHOD A: fixed-duration clip (press 'r', stops by itself after CLIP_DURATION_SEC) >>>
def on_key_press(event):
    """Matplotlib key-press handler: press 'r' to start a fixed-duration recording."""
    global is_recording, record_until, rgb_buffer, depth_color_buffer, depth_raw_buffer

    if event.key == 'r':
        with recording_lock:
            if is_recording:
                print("[Recording] Already recording — ignoring keypress.")
                return
            is_recording = True
            record_until = time.time() + CLIP_DURATION_SEC
            rgb_buffer = []
            depth_color_buffer = []
            depth_raw_buffer = []
            print(f"[Recording] Started — capturing {CLIP_DURATION_SEC}s clip …")


KEY_HANDLERS.append(on_key_press)
KEY_HINTS.append("'r' = record a clip")
STARTUP_INFO.append(f"Press 'r' in the figure window to record a {CLIP_DURATION_SEC}s clip.")
# <<< METHOD A <<<


# >>> METHOD B: manual start / stop (press 'b' to begin, 'e' to end) >>>
# Keys are chosen to avoid matplotlib's built-in shortcuts in the figure
# window (e.g. 's' opens a save-file dialog that freezes the capture loop,
# 'f' = fullscreen, 'q' = quit, 'p'/'o' = pan/zoom, 'g'/'k'/'l' = grid/scales).
START_KEY = 'b'
STOP_KEY = 'e'
# Safety cap: frames are held in memory until the clip is saved (roughly
# 17 MB per second at ~10 fps), so a recording you forget to stop is ended
# and saved automatically after this many seconds. Raise it if you need longer.
MANUAL_MAX_SEC = 60


def on_key_press_manual(event):
    """Matplotlib key-press handler: START_KEY begins recording, STOP_KEY ends it and saves the clip."""
    global is_recording, record_until, rgb_buffer, depth_color_buffer, depth_raw_buffer

    if event.key == START_KEY:
        with recording_lock:
            if is_recording:
                print(f"[Recording] Already recording — press '{STOP_KEY}' to stop.")
                return
            is_recording = True
            record_until = time.time() + MANUAL_MAX_SEC   # safety cap only; normally you stop it
            rgb_buffer = []
            depth_color_buffer = []
            depth_raw_buffer = []
            print(f"[Recording] Started — press '{STOP_KEY}' to stop "
                  f"(auto-stops after {MANUAL_MAX_SEC}s).")

    elif event.key == STOP_KEY:
        with recording_lock:
            if not is_recording:
                print(f"[Recording] Not recording — press '{START_KEY}' to start "
                      f"(a clip that hit the {MANUAL_MAX_SEC}s cap has already been saved).")
                return
            is_recording = False
            n_frames = len(rgb_buffer)
            if n_frames == 0:
                print("[Recording] Stopped, but no frames were captured — nothing saved.")
                return
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            t = threading.Thread(
                target=save_clip,
                args=(list(rgb_buffer), list(depth_color_buffer), list(depth_raw_buffer), timestamp),
                daemon=True,
            )
            t.start()
            print(f"[Recording] Stopped — {n_frames} frames captured "
                  f"(~{n_frames / ESTIMATED_FPS:.0f}s), saving in background.")


KEY_HANDLERS.append(on_key_press_manual)
KEY_HINTS.append(f"'{START_KEY}' = start, '{STOP_KEY}' = stop")
STARTUP_INFO.append(f"Press '{START_KEY}' in the figure window to start recording and '{STOP_KEY}' to stop.")
# <<< METHOD B <<<


def parse_frame(frame_data: bytes):
    """Decode a raw HTTP frame into (depth_mapped, depth_color_bgr, rgb) numpy arrays."""
    config = frame_config_decode(frame_data[16:16+12])
    frame_bytes = frame_payload_decode(frame_data[16+12:], config)

    depth = np.frombuffer(frame_bytes[0], 'uint16' if 0 == config[1] else 'uint8').reshape(
        240, 320) if frame_bytes[0] else None

    rgb = np.frombuffer(frame_bytes[3], 'uint8').reshape(
        (480, 640, 3)) if frame_bytes[3] else None

    depth_mapped = None
    depth_color = None
    if depth is not None:
        depth_f = depth.astype(float)
        depth_mapped = ((depth_f - DEPTH_OLD_MIN) * (DEPTH_NEW_MAX - DEPTH_NEW_MIN)) / \
                       (DEPTH_OLD_MAX - DEPTH_OLD_MIN) + DEPTH_NEW_MIN
        depth_color = depth_to_colormap(depth_mapped)

    return depth_mapped, depth_color, rgb, config


def show_frame(fig, frame_data: bytes):
    depth_mapped, depth_color, rgb, config = parse_frame(frame_data)

    # --- Buffer frames if recording ---
    with recording_lock:
        global is_recording
        if is_recording:
            now = time.time()
            if now < record_until:
                if rgb is not None:
                    rgb_buffer.append(rgb.copy())
                if depth_color is not None:
                    depth_color_buffer.append(depth_color.copy())
                if depth_mapped is not None:
                    depth_raw_buffer.append(depth_mapped.copy())
            else:
                # Recording window has ended — save in background
                is_recording = False
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                t = threading.Thread(
                    target=save_clip,
                    args=(list(rgb_buffer), list(depth_color_buffer), list(depth_raw_buffer), timestamp),
                    daemon=True,
                )
                t.start()
                print("[Recording] Clip complete — saving in background.")

    # --- Display ---
    ax1 = fig.add_subplot(121)
    if depth_mapped is not None:
        cax1 = ax1.imshow(depth_mapped, cmap='jet_r')
        ax1.set_xticks([])
        ax1.set_yticks([])
        rec_label = "  [REC]" if is_recording else ""
        ax1.set_title(f"Depth Map (150mm-1500mm), (240x320 pixels){rec_label}")
        fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

    ax4 = fig.add_subplot(122)
    if rgb is not None:
        ax4.imshow(rgb)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("RGB image (480x640 pixels)  |  " + "   ".join(KEY_HINTS))


def connect_to_camera(max_attempts=10, retry_delay=2.0):
    """Retry initial configuration instead of giving up (or crashing)
    the instant the camera doesn't answer on the first try — it can take
    a few seconds after being plugged in / powered on for its HTTP
    server to come up."""
    for attempt in range(1, max_attempts + 1):
        if post_encode_config(frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)):
            return True
        print(f"[Info] Camera not reachable at {HOST}:{PORT} "
              f"(attempt {attempt}/{max_attempts}) - retrying in {retry_delay:.0f}s...")
        time.sleep(retry_delay)
    return False


if connect_to_camera():
    plt.ion()
    figsize = (12, 6)
    fig = plt.figure('2D frame', figsize=figsize)
    for handler in KEY_HANDLERS:                               # register the enabled recording method(s)
        fig.canvas.mpl_connect('key_press_event', handler)

    for line in STARTUP_INFO:
        print(f"[Info] {line}")
    print(f"[Info] Clips will be saved to: {os.path.abspath(OUTPUT_DIR)}/")

    consecutive_failures = 0
    while True:
        p = get_frame_from_http()
        if p is not None:
            consecutive_failures = 0
            show_frame(fig, p)
        else:
            consecutive_failures += 1
            if consecutive_failures == 1:
                print(f"[Warn] Lost contact with camera at {HOST}:{PORT} - will keep "
                      f"retrying (check the USB-C connection if this doesn't recover).")
            elif consecutive_failures % 20 == 0:
                print(f"[Warn] Still no response from camera after {consecutive_failures} attempts...")
            time.sleep(0.2)
        plt.pause(0.001)
        fig.clf()
else:
    print(f"[Error] Could not configure camera at {HOST}:{PORT} after several attempts.")
    print("        Check that: the camera is connected via USB-C and powered on,")
    print("        Windows shows a network adapter on the 192.168.233.x subnet")
    print("        (Settings > Network, or 'ipconfig' in cmd), and that no other")
    print("        program (e.g. gui.py) already has the camera open. Then re-run.")
