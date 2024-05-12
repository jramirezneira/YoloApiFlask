

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
# import base64
# import asyncio
# from flask_sock import Sock
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
import psutil




# queue_url = 'https://sqs.us-west-2.amazonaws.com/156581257326/yolo'
# bucket_name='variosjavierramirez'
#
#
# session = boto3.session.Session(profile_name="default")
# sqs = session.client('sqs', verify=False)
# s3_resource = session.resource('s3', verify=False)
# bucket = s3_resource.Bucket(bucket_name)
# s3location = s3_resource.get_bucket_location(Bucket=bucket_name)['LocationConstraint']

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


@sock.route('/detect')
# @app.websocket('/detect')
def video_feed(sock):

    

    opt.source = sock.receive()
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

    # bytetracker = BYTETracker(
    #     track_thresh=0.6, match_thresh=0.8, track_buffer=120, frame_rate=30
    # )
    # tracker = bytetracker


    # Define region points
    #region_points = [(400, 920), (920, 720), (920, 80), (400, 80)]
    region_points = [(0,400), (1280, 400), (1280, 0), (0,0)]


    # Init Object Counter
    counter = object_counter.ObjectCounter()
    counter.set_args(view_img=False,
                    reg_pts=region_points,
                    classes_names=names,
                    draw_tracks=True)

    conf_thres = 0.25
    iou_thres = 0.45
    classes = None
    agnostic_nms = False
    max_det = 1000
    line_thickness = 2
    half = False
    # imgsz = (640, 640)
    vid_stride = 50


    source = str(source)
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
            #         if obj.cap.isOpened():
                    obj.cap.release() 
                    obj.cv2.destroyAllWindows()
                    LOGGER.info("close release objet {obj}")
                except:
                    LOGGER.error("An exception occurred in obj.cap.release {obj}")
                # for thr in obj.threads:
                #     thr.join(1)


        dataset =LoadStreams(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        # dataset = LoadImages("istockphoto-899486586-640_adpp_is.mp4", imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, imgsz=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, imgsz=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    

    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch

        if frame_idx%1000==0:
            im0=im0s[0].copy()
            cv2.imwrite(save_dir / f'{frame_idx}.jpg', im0) 
            print(save_dir  / f'{frame_idx}.jpg')


        



  




@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/loading_image')
def get_image():    
    return send_file("loading.gif", mimetype='image/gif')


if __name__ == "__main__":
    # Input arguments
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
    parser.add_argument('--vid-stride', type=int, default=10, help='video frame-rate stride')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # Just in case one dimension if provided, i.e., if 640 is provided, image inference will be over 640x640 images
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    # Check requirements are installed
    check_requirements(requirements=ROOT.parent / 'requirements.txt',exclude=('tensorboard', 'thop'))

    # Load model
    # opt.device = select_device(opt.device)
    # model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16 = opt.half)
    # stride, names, pt = model.stride, model.names, model.pt
    # model = AutoBackend("yolov8n.pt")
    # model.warmup()
    # stride, names, pt = model.stride, model.names, model.pt

    stride=32
    names = {   0: 'person',
  1: 'bicycle',
  2: 'car',
  3: 'motorcycle',
  4: 'airplane',
  5: 'bus',
  6: 'train',
  7: 'truck',
  8: 'boat',
  9: 'traffic light',
  10: 'fire hydrant',
  11: 'stop sign',
  12: 'parking meter',
  13: 'bench',
  14: 'bird',
  15: 'cat',
  16: 'dog',
  17: 'horse',
  18: 'sheep',
  19: 'cow',
  20: 'elephant',
  21: 'bear',
  22: 'zebra',
  23: 'giraffe',
  24: 'backpack',
  25: 'umbrella',
  26: 'handbag',
  27: 'tie',
  28: 'suitcase',
  29: 'frisbee',
  30: 'skis',
  31: 'snowboard',
  32: 'sports ball',
  33: 'kite',
  34: 'baseball bat',
  35: 'baseball glove',
  36: 'skateboard',
  37: 'surfboard',
  38: 'tennis racket',
  39: 'bottle',
  40: 'wine glass',
  41: 'cup',
  42: 'fork',
  43: 'knife',
  44: 'spoon',
  45: 'bowl',
  46: 'banana',
  47: 'apple',
  48: 'sandwich',
  49: 'orange',
  50: 'broccoli',
  51: 'carrot',
  52: 'hot dog',
  53: 'pizza',
  54: 'donut',
  55: 'cake',
  56: 'chair',
  57: 'couch',
  58: 'potted plant',
  59: 'bed',
  60: 'dining table',
  61: 'toilet',
  62: 'tv',
  63: 'laptop',
  64: 'mouse',
  65: 'remote',
  66: 'keyboard',
  67: 'cell phone',
  68: 'microwave',
  69: 'oven',
  70: 'toaster',
  71: 'sink',
  72: 'refrigerator',
  73: 'book',
  74: 'clock',
  75: 'vase',
  76: 'scissors',
  77: 'teddy bear',
  78: 'hair drier',
  79: 'toothbrush'}
    pt= True
    model = YOLO("yolov8s.pt") 
    # stride, names, pt = model.stride, model.names, model.pt


    # loadStream = LoadStreams(imgsz=opt.imgsz, stride=stride, auto=pt, vid_stride=opt.vid_stride)

    # detect(opt)

    # Run app


    app.run(host="0.0.0.0", port=opt.port, debug=True) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)

   
