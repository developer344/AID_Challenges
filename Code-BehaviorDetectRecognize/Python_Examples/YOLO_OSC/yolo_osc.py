"""Runs YOLO on the webcam and sends every detection to Processing by OSC.

    pip install ultralytics python-osc
    python yolo_osc.py                 # the pretrained model (80 COCO classes), camera 0
    python yolo_osc.py runs/detect/train/weights/best.pt    # your fine-tuned model
    python yolo_osc.py yolo11n.pt 1    # another camera (e.g. 1 if 0 is the iPhone's Continuity Camera)
    python yolo_osc.py yolo11n.pt 0 show    # also open a preview window with the boxes

The first time, macOS asks to let the terminal use the camera; if the first run fails, run it again.
Messages, to 127.0.0.1:12000:
    /yolo/frame  n                                  at the start of each frame: n detections follow
    /yolo/box    name confidence x1 y1 x2 y2        one per object; corners from 0 to 1 of the image
Then run the YOLO_Receiver sketch. Once a second the script prints what it sees.
"""
import sys
import time

from pythonosc.udp_client import SimpleUDPClient
from ultralytics import YOLO

model = YOLO(sys.argv[1] if len(sys.argv) > 1 else "yolo11n.pt")
camera = int(sys.argv[2]) if len(sys.argv) > 2 else 0
show = len(sys.argv) > 3 and sys.argv[3] == "show"
osc = SimpleUDPClient("127.0.0.1", 12000)

last_print, frames = time.time(), 0
for result in model.predict(source=camera, stream=True, verbose=False, show=show):
    osc.send_message("/yolo/frame", len(result.boxes))
    names = []
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxyn[0].tolist()
        name = model.names[int(box.cls)]
        names.append(name)
        osc.send_message("/yolo/box", [name, float(box.conf), x1, y1, x2, y2])
    frames += 1
    if time.time() - last_print >= 1:
        print(f"{frames / (time.time() - last_print):.0f} frames/s, sending: {', '.join(names) or 'nothing detected'}")
        last_print, frames = time.time(), 0
