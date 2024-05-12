# import required libraries
from vidgear.gears import NetGear
import cv2
import pafy
import os
import json
from flask import Flask, jsonify, request, Response
from flask_cors import CORS, cross_origin
import gc
from ultralytics.utils.ops import LOGGER
import threading
from ultralytics import YOLO
from ultralytics.solutions import object_counter
from utils.stream_loaders import LoadStreams, LoadImages, LoadStreamNoThread
import base64
from collections import  Counter
import numpy as np
import time
import websockets
import asyncio


model = YOLO("yolov8n.pt") 
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

# region_points = [{    
#                 "source":"https://www.youtube.com/watch?v=5_XSYlAfJZM",
#                  "region_points":[[(800,600), (1280, 600), (1280, 0), (800,0)], [(0,300), (500, 300), (500, 720), (0,720)]],
#                  "reg_counts":[[(800,600), (1000,700)], [(0,200),(200,300)]]                 
#                  },
#                  {    
#                 "source":"https://www.youtube.com/watch?v=MNn9qKG2UFI",
#                  "region_points":[[(800,600), (1280, 600), (1280, 0), (800,0)], [(0,300), (500, 300), (500, 720), (0,720)]],
#                  "reg_counts":[[(800,600), (1000,700)], [(0,200),(200,300)]]                                  
#                  },
#                  {    
#                 "source":"https://www.youtube.com/watch?v=PtChZ0D7tkE",
#                  "region_points":[[(800,600), (1280, 600), (1280, 0), (800,0)], [(0,300), (500, 300), (500, 720), (0,720)]],
#                  "reg_counts":[[(800,600), (1000,700)], [(0,200),(200,300)]]                                  
#                  }]


# reg_counts=[[(800,600), (1000,700)], [(0,200),(200,300)]]


video, best, cap = None, None, None


# activate multiclient_mode mode
options = {"multiclient_mode": True}


app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


server = NetGear(
    address="127.0.0.1",
    port=["5567"],
    protocol="tcp",
    pattern=2,
    logging=True,
    **options
)


def getStatus():
    with open("app.conf",  "r") as json_data_file:
        return json.load(json_data_file)["statusServer"]

def setStatus(status):    
    with open("app.conf",  "r") as json_data_file:
        data = json.load(json_data_file)
        data["statusServer"] = status
        newData=json.dumps(data)

    with open("app.conf", 'w') as file:
        file.write(newData)
    return status
setStatus("offline")



@app.route('/api/region_points', methods=['GET'])
@cross_origin()
def region_points():
    with open("app.conf",  "r") as json_data_file:
        return json.load(json_data_file)["region_points"]
    

@app.route('/api/status', methods=['GET'])
@cross_origin()
def status():
    response = {'message': getStatus()}
    return jsonify(response)

@app.route('/api/start', methods=['GET'])
@cross_origin()
def start():
    url=request.args.get('url')
    response = {'message': setStatus('active')}
    thr = threading.Thread(target=service, args=([url]), kwargs={})
    thr.start()
    
    return jsonify(response)


@app.route('/api/stop', methods=['GET'])
@cross_origin()
def stop():
    for obj in gc.get_objects():
        if isinstance(obj, LoadStreamNoThread):
            try:
                obj.cap.release()                 
                LOGGER.info("close release objet {obj}")
            except:
                LOGGER.error("An exception occurred in obj.cap.release {obj}")

    response = {'message': setStatus('offline')}
    return jsonify(response)


# Open suitable video stream (webcam on first index in our case)
# Define received data dictionary
data_dict = {}

# loop over until KeyBoard Interrupted

def service(source, isStreaming=True):
    # cont=0
    counter=[]
    vid_writer=None
    with open("app.conf",  "r") as json_data_file:
        region_points= json.load(json_data_file)["region_points"]

    region_points_dict = [x for x in region_points if x['source'] == source][0]

    for i, rp in enumerate(region_points_dict["region_points"]):
        ctr= object_counter.ObjectCounter()
        ctr.set_args(view_img=False,
                    reg_pts=rp,
                    classes_names=names,
                    draw_tracks=True,
                    reg_counts=region_points_dict["reg_counts"][i]
                    )
        counter.append(ctr)
    
    if isStreaming:
        # dataset =LoadStreams(source, imgsz=[288, 480], auto=True, vid_stride=1)        
        try:
            ldst = LoadStreamNoThread(source)
            cap = ldst.getCap()
        except:
            LOGGER.error("An exception occurred to open cap.release")
    else:
        dataset = LoadImages(source, imgsz=[288, 480], stride=32, auto=True, vid_stride=1)

    # for frame_idx, batch in enumerate(dataset):
        # 
    n=0
    while cap.isOpened():
        try:
            n += 1          
            success, im0 = cap.read()  
            if n % 1== 0:
                dict_result=dict()
                dict_result["verbose"] =False
                results = model.track(im0, persist=True, imgsz=480, show=False, **dict_result)

                for ctr in counter:
                    im0 = ctr.start_counting(im0, results)    
                
                # Convert list to an array
                time.sleep(0.001)
                server.send(im0)        
        except:
            LOGGER.error("sale del while true")
            cv2.destroyAllWindows()
            setStatus("offline")
            break 

    # safely close video stream
    cv2.destroyAllWindows()
    # safely close server
    # server.close()
    setStatus("offline")


@app.route('/detect',methods = ['POST', 'GET'])
def video_feed():
    """Video streaming home page."""
    source = request.args.get('url')

    return Response(detect(source), mimetype='multipart/x-mixed-replace; boundary=frame')

def detect(source):
    dataset = LoadStreams(source, imgsz=[288, 480], stride=32, auto=True, vid_stride=1)
    for frame_idx, batch in enumerate(dataset):
        path, im, im0s, vid_cap, s = batch
        im0=im0s[0].copy()
        results = model.track(im0, persist=True, show=False)

       
        im0 = cv2.imencode('.jpg', im0)[1].tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True,  port=5001)







    
