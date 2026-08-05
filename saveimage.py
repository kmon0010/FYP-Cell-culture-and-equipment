from PIL import Image
import requests
import matplotlib.pyplot as plt
import struct
import numpy as np
import cv2
import time

# Global variables
latest_rgb = None
screenshot_count = 0


def frame_config_decode(frame_config):
    return struct.unpack("<BBBBBBBBi", frame_config)


def frame_config_encode(trigger_mode=1, deep_mode=1, deep_shift=255,
                        ir_mode=1, status_mode=2, status_mask=7,
                        rgb_mode=1, rgb_res=0, expose_time=0):

    return struct.pack("<BBBBBBBBi",
                       trigger_mode, deep_mode, deep_shift,
                       ir_mode, status_mode, status_mask,
                       rgb_mode, rgb_res, expose_time)


def frame_payload_decode(frame_data: bytes, with_config: tuple):

    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]

    deepth_size = (320 * 240 * 2) >> with_config[1]

    deepth_img = struct.unpack(
        "<%us" % deepth_size,
        frame_payload[:deepth_size]
    )[0] if 0 != deepth_size else None

    frame_payload = frame_payload[deepth_size:]

    ir_size = (320 * 240 * 2) >> with_config[3]

    ir_img = struct.unpack(
        "<%us" % ir_size,
        frame_payload[:ir_size]
    )[0] if 0 != ir_size else None

    frame_payload = frame_payload[ir_size:]

    status_size = (320 * 240 // 8) * (
        16 if 0 == with_config[4]
        else 2 if 1 == with_config[4]
        else 8 if 2 == with_config[4]
        else 1
    )

    status_img = struct.unpack(
        "<%us" % status_size,
        frame_payload[:status_size]
    )[0] if 0 != status_size else None

    frame_payload = frame_payload[status_size:]

    assert(deep_data_size == deepth_size + ir_size + status_size)

    rgb_size = len(frame_payload)

    assert(rgb_data_size == rgb_size)

    rgb_img = struct.unpack(
        "<%us" % rgb_size,
        frame_payload[:rgb_size]
    )[0] if 0 != rgb_size else None

    if (rgb_img is not None) and (1 == with_config[6]):

        jpeg = cv2.imdecode(
            np.frombuffer(rgb_img, dtype='uint8'),
            cv2.IMREAD_COLOR
        )

        if jpeg is not None:
            rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
            rgb_img = rgb.tobytes()
        else:
            rgb_img = None

    return (deepth_img, ir_img, status_img, rgb_img)


HOST = '192.168.233.1'
PORT = 80


def post_encode_config(config=frame_config_encode(),
                       host=HOST,
                       port=PORT):

    r = requests.post(
        'http://{}:{}/set_cfg'.format(host, port),
        config
    )

    return r.status_code == requests.codes.ok


def get_frame_from_http(host=HOST, port=PORT):

    r = requests.get(
        'http://{}:{}/getdeep'.format(host, port)
    )

    if r.status_code == requests.codes.ok:
        return r.content


def on_key(event):
    global latest_rgb
    global screenshot_count

    if event.key == 'r':

        if latest_rgb is not None:

            filename = f"rgb_capture_{screenshot_count}.png"

            image = Image.fromarray(latest_rgb)
            image.save(filename)

            print(f"Saved screenshot: {filename}")

            screenshot_count += 1

        else:
            print("No RGB frame available yet.")


def show_frame(fig, frame_data: bytes):

    global latest_rgb

    config = frame_config_decode(frame_data[16:16+12])

    frame_bytes = frame_payload_decode(
        frame_data[16+12:],
        config
    )

    depth = np.frombuffer(
        frame_bytes[0],
        'uint16' if 0 == config[1] else 'uint8'
    ).reshape(240, 320) if frame_bytes[0] else None

    rgb = np.frombuffer(
        frame_bytes[3],
        'uint8'
    ).reshape((480, 640, 3)) if frame_bytes[3] else None

    # Store latest RGB frame for screenshots
    latest_rgb = rgb

    old_min = 16
    old_max = 255
    new_min = 150
    new_max = 1500

    depth = depth.astype(float)

    mapped_depth = (
        ((depth - old_min) * (new_max - new_min))
        / (old_max - old_min)
    ) + new_min

    ax1 = fig.add_subplot(121)

    cax1 = ax1.imshow(mapped_depth, cmap='jet_r')

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title("Depth Map")

    ax4 = fig.add_subplot(122)

    if rgb is not None:
        ax4.imshow(rgb)

    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_title("RGB image")

    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)


if post_encode_config(frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)):

    plt.ion()

    figsize = (12, 12)

    fig = plt.figure('2D frame', figsize=figsize)

    # Connect keyboard event
    fig.canvas.mpl_connect('key_press_event', on_key)

    while True:

        p = get_frame_from_http()

        show_frame(fig, p)

        plt.pause(0.001)

        fig.clf()

    plt.ioff()