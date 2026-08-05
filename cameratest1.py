from PIL import Image  # imaging library
import requests # http library
import matplotlib.pyplot as plt
import struct
import numpy as np
import cv2


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


def post_encode_config(config=frame_config_encode(), host=HOST, port=PORT):
    r = requests.post('http://{}:{}/set_cfg'.format(host, port), config)
    if(r.status_code == requests.codes.ok):
        return True
    return False


def get_frame_from_http(host=HOST, port=PORT):
    r = requests.get('http://{}:{}/getdeep'.format(host, port))
    if(r.status_code == requests.codes.ok):
        # print('Get deep image')
        deepimg = r.content
        # print('Length={}'.format(len(deepimg)))
        (frameid, stamp_msec) = struct.unpack('<QQ', deepimg[0:8+8])
        # print((frameid, stamp_msec/1000))
        return deepimg


def show_frame(fig, frame_data: bytes):
    config = frame_config_decode(frame_data[16:16+12])
    frame_bytes = frame_payload_decode(frame_data[16+12:], config)

    depth = np.frombuffer(frame_bytes[0], 'uint16' if 0 == config[1] else 'uint8').reshape(
        240, 320) if frame_bytes[0] else None

    ir = np.frombuffer(frame_bytes[1], 'uint16' if 0 == config[3] else 'uint8').reshape(
        240, 320) if frame_bytes[1] else None

    status = np.frombuffer(frame_bytes[2], 'uint16' if 0 == config[4] else 'uint8').reshape(
        240, 320) if frame_bytes[2] else None

    rgb = np.frombuffer(frame_bytes[3], 'uint8').reshape(
        (480, 640, 3)) if frame_bytes[3] else None

    '''

    # Print min and max values
    print("Min depth value:", np.min(depth))
    print("Max depth value:", np.max(depth))

    '''

    old_min = 16
    old_max = 255
    new_min = 150
    new_max = 1500

    # Map the entire array from [old_min, old_max] to [new_min, new_max]
    depth = depth.astype(float)
    mapped_depth = ((depth - old_min) * (new_max - new_min)) / (old_max - old_min) + new_min

    ax1 = fig.add_subplot(121)
    cax1 = ax1.imshow(mapped_depth, cmap='jet_r')
    
    if not depth is None:

        ax1.imshow(mapped_depth, cmap='jet_r')
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("Depth Map (150mm-1500mm), (240x320 pixels)")


    ax4 = fig.add_subplot(122)
    if not rgb is None:
        ax4.imshow(rgb)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.set_title("RGB image (1200x1600 pixels)")

    fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)

if post_encode_config(frame_config_encode(1, 1, 255, 0, 2, 7, 1, 0, 0)):
     plt.ion()
     figsize = (12, 12)
     fig = plt.figure('2D frame', figsize=figsize)
     while True:
         p = get_frame_from_http()
         show_frame(fig, p)
         plt.pause(0.001)
         fig.clf()

     plt.ioff()
# import requests

# host = "192.168.233.1"

# r = requests.get(f"http://{host}")
# print(r.status_code)