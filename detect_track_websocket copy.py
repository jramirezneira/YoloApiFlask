# YOLOv8API flask for object detection

"""
Run YOLOv8 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.
Usage - sources:
    $ python detect_api.py --weights yolov8s.pt --source 0                               # webcam
                                                         img.jpg                         # image
                                                         vid.mp4                         # video
                                                         screen                          # screenshot
                                                         path/                           # directory
                                                         list.txt                        # list of images
                                                         list.streams                    # list of streams
                                                         'path/*.jpg'                    # glob
                                                         'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                         'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream
Usage - formats:
    $ python detect_api.py --weights yolov8s.pt                 # PyTorch
                                     yolov8s.torchscript        # TorchScript
                                     yolov8s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                     yolov8s_openvino_model     # OpenVINO
                                     yolov8s.engine             # TensorRT
                                     yolov8s.mlmodel            # CoreML (macOS-only)
                                     yolov8s_saved_model        # TensorFlow SavedModel
                                     yolov8s.pb                 # TensorFlow GraphDef
                                     yolov8s.tflite             # TensorFlow Lite
                                     yolov8s_edgetpu.tflite     # TensorFlow Edge TPU
                                     yolov8s_paddle_model       # PaddlePaddle
"""

# ========== Libraries for the API ============= #

from argparse import ArgumentParser

import cv2
import numpy as np


from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils.plotting import Annotator, colors
import torch
from bytetrack.byte_tracker import BYTETracker
# from ultralytics.yolo.data.dataloaders.stream_loaders import LoadImages, LoadStreams, LoadScreenshots
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from utils.stream_loaders import LoadStreams, LoadImages, LoadScreenshots


from flask import Flask, render_template, Response, request, send_file
import json
import pandas as pd
from PIL import Image, ImageFont


# ========== Libraries from detect.py ============= #

import argparse
import os
import platform
import sys
from pathlib import Path
import time
import torch
import base64


font_size = 28
font_filepath = "arial.ttf"
color = (67, 33, 116, 155)

font = ImageFont.truetype(font_filepath, size=font_size)


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv8API root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from ultralytics.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.engine.predictor import AutoBackend as DetectMultiBackend
# from ultralytics.data.v5loader import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.utils.ops import LOGGER, Profile, non_max_suppression, scale_boxes, xyxy2xywh
from ultralytics.utils.checks import check_file, check_imgsz, check_requirements, colorstr, cv2, print_args
from ultralytics.utils.files import increment_path
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import strip_optimizer, select_device, smart_inference_mode


from ultralytics import YOLO
from ultralytics.solutions import object_counter

# Extra utils
from utils.general import update_options, image_resize
import numpy as np

# from ultralytics.trackers.track import BYTETracker
# from bytetrack.byte_tracker import BYTETracker
from flask_sock import Sock
from argparse import Namespace
import gc
import asyncio
# import websockets

# Initialize flask API
app = Flask(__name__)
sock = Sock(app)
dataset= None

n = 1

imgs, fps, frames, threads, cap = [None] * n, [0] * n, [0] * n, [None] * n, [None] * n


parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov8s.pt', help='model path or triton URL')
parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[288, 480], help='inference size h,w')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='show results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--visualize', action='store_true', help='visualize features')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
parser.add_argument('--vid-stride', type=int, default=3, help='video frame-rate stride')
parser.add_argument('--port', default=5000, type=int, help='port deployment')
opt, unknown = parser.parse_known_args()


pt= True
model = YOLO("yolov8n.pt") 

# Just in case one dimension if provided, i.e., if 640 is provided, image inference will be over 640x640 images
opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
print_args(vars(opt))

# Check requirements are installed
check_requirements(requirements=ROOT.parent / 'requirements.txt',exclude=('tensorboard', 'thop'))



stride=32
names = {   0: 'person',  1: 'bicycle',  2: 'car',  3: 'motorcycle',  4: 'airplane',  5: 'bus',  6: 'train',  7: 'truck',
  8: 'boat',  9: 'traffic light',  10: 'fire hydrant',  11: 'stop sign',  12: 'parking meter',  13: 'bench',  14: 'bird',
  15: 'cat',  16: 'dog',  17: 'horse',  18: 'sheep',  19: 'cow',  20: 'elephant',  21: 'bear',  22: 'zebra',  23: 'giraffe',
  24: 'backpack',  25: 'umbrella',  26: 'handbag',  27: 'tie',  28: 'suitcase',  29: 'frisbee',  30: 'skis',  31: 'snwboard',
  32: 'sports ball',  33: 'kite',  34: 'baseball bat',  35: 'baseball glove',  36: 'skateboard',  37: 'surfboard',  38: 'tennis racket',
  39: 'bottle',  40: 'wine glass',  41: 'cup',  42: 'fork',  43: 'knife',  44: 'spoon',  45: 'bowl',  46: 'banana',  47: 'apple',
  48: 'sandwich',  49: 'orange',  50: 'broccoli',  51: 'carrot',  52: 'hot dog',  53: 'pizza',  54: 'donut',  55: 'cake',
  56: 'chair',  57: 'couch',  58: 'potted plant',  59: 'bed',  60: 'dining table',  61: 'toilet',  62: 'tv',  63: 'laptop',
  64: 'mouse',  65: 'remote',  66: 'keyboard',  67: 'cell phone',  68: 'microwave',  69: 'oven',  70: 'toaster',  71: 'sink',
  72: 'refrigerator',  73: 'book',  74: 'clock',  75: 'vase',  76: 'scissors',  77: 'teddy bear',  78: 'hair drier',  79: 'toothbrush'}


region_points = [(800,600), (1280, 600), (1280, 0), (800,0)]
region_points2 = [(0,300), (500, 300), (500, 720), (0,720)]

# Init Object Counter
counter = object_counter.ObjectCounter()
# counter.set_args(view_img=False,
#                 reg_pts=region_points,
#                 classes_names=names,
#                 draw_tracks=True)

counter2 = object_counter.ObjectCounter()
# counter2.set_args(view_img=False,
#                 reg_pts=region_points2,
#                 classes_names=names,
#                 draw_tracks=True)

opt.source =""
weights=opt.weights  # model path or triton URL
source=opt.source  # file/dir/URL/glob/screen/0(webcam)
imgsz=opt.imgsz  # inference size (height, width)
conf_thres=opt.conf_thres  # confidence threshold
iou_thres=opt.iou_thres  # NMS IOU threshold
max_det=opt.max_det  # maximum detections per image
view_img=opt.view_img  # show results
save_txt=opt.save_txt  # save results to *.txt
save_conf=False  # save confidences in --save-txt labels
save_crop=opt.save_crop  # save cropped prediction boxes
nosave=opt.nosave  # do not save images/videos
classes=opt.classes  # filter by class: --class 0, or --class 0 2 3
agnostic_nms=opt.agnostic_nms  # class-agnostic NMS
augment=opt.augment  # augmented inference
visualize=opt.visualize  # visualize features
update=opt.update  # update all models
project=opt.project  # save results to project/name
name=opt.name  # save results to project/name
exist_ok=opt.exist_ok  # existing project/name ok, do not increment
line_thickness=opt.line_thickness  # bounding box thickness (pixels)
hide_labels=opt.hide_labels  # hide labels
hide_conf=opt.hide_conf  # hide confidences
vid_stride=opt.vid_stride  # video frame-rate stride


source="https://www.youtube.com/watch?v=bJpxhD_JDAA"
save_img = not nosave and not source.endswith('.txt')  # save inference images
is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
screenshot = source.lower().startswith('screen')
if is_url and is_file:
    source = check_file(source)  # download

# Directories
save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

# Load model (outside of the function)
imgsz = check_imgsz(imgsz, stride=stride)  # check image size

# cleanup_stop_thread()
# sys.exit()
# Dataloader
bs = 1  # batch_size

if webcam:
    # dataset = loadStream.newThreads(source)
    for obj in gc.get_objects():
        if isinstance(obj, LoadStreams):
            try:
                obj.cap.release() 
                obj.cv2.destroyAllWindows()
                LOGGER.info("close release objet {obj}")
            except:
                LOGGER.error("An exception occurred in obj.cap.release {obj}")


    dataset =LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    # dataset = LoadImages("istockphoto-899486586-640_adpp_is.mp4", imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    bs = len(dataset)
elif screenshot:
    dataset = LoadScreenshots(source, imgsz=imgsz, stride=stride, auto=pt)
else:
    dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)




def update(self, i,  stream, timewait):
    """Read stream `i` frames in daemon thread."""
        
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        im0=im0s[0].copy()

        results = model.track(im0, persist=True, show=False)

        im0 = counter.start_counting(im0, results)
        im0 = counter2.start_counting(im0, results)
        im0 = cv2.imencode('.jpg', im0)[1].tobytes()
        dict_result=dict()
        dict_result["img"] =base64.b64encode(im0).decode('utf-8')
    # sock.send(json.dumps(dict_result))
        
        

@sock.route('/detect')
# @app.websocket('/detect')
def video_feed(sock):   

    opt.source = sock.receive()
    source = str(source)
    

  




@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/loading_image')
def get_image():    
    return send_file("loading.gif", mimetype='image/gif')


# if __name__ == "__main__":
    # Input arguments


    # Load model
    # opt.device = select_device(opt.device)
    # model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16 = opt.half)
    # stride, names, pt = model.stride, model.names, model.pt
    # model = AutoBackend("yolov8n.pt")
    # model.warmup()
    # stride, names, pt = model.stride, model.names, model.pt



    # stride, names, pt = model.stride, model.names, model.pt


    # loadStream = LoadStreams(imgsz=opt.imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)

    # detect(opt)

    # Run app


app.run(host="0.0.0.0", port=opt.port, debug=True) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)

   
