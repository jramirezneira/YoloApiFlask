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

import torch
import boto3




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


from ultralytics.yolo.engine.predictor import AutoBackend as DetectMultiBackend
from ultralytics.yolo.data.utils import IMG_FORMATS, VID_FORMATS
from ultralytics.yolo.data.dataloaders.v5loader import LoadImages, LoadScreenshots, LoadStreams
from ultralytics.yolo.utils.ops import LOGGER, Profile, non_max_suppression, scale_boxes, xyxy2xywh
from ultralytics.yolo.utils.checks import check_file, check_imgsz, check_requirements, colorstr, cv2, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from ultralytics.yolo.utils.torch_utils import strip_optimizer, select_device, smart_inference_mode

# Extra utils
from utils.general import update_options
import numpy as np

# Initialize flask API
app = Flask(__name__)
# sock = Sock(app)

@smart_inference_mode()
def detect(opt):
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

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s +=''  #f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '' #'%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # Stream results
            # im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'image':
            #         cv2.imwrite(save_path, im0)
            #     else:  # 'video' or 'stream'
            #         if vid_path[i] != save_path:  # new video
            #             vid_path[i] = save_path
            #             if isinstance(vid_writer[i], cv2.VideoWriter):
            #                 vid_writer[i].release()  # release previous video writer
            #             if vid_cap:  # video
            #                 fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                 w = 144#int(imgsz[1]) # int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                 h = 96#int(imgsz[0]) #int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             else:  # stream
            #                 fps, w, h = 5, im0.shape[1], im0.shape[0]
            #             save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #             vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #         vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")
      
        # This is done in order to be shown in a browser, save_txt will return json file, otherwise, an image in bytes
        if save_txt:
            if os.path.exists(txt_path + '.txt'):
                result = pd.read_csv(txt_path + '.txt', sep =" ",names = ["class","x","y","w","h","conf"], header = None)
                result = result.to_json(orient="records")
                result = json.loads(result)
                
            else:
                result = []

            # im0 = cv2.imencode('.jpg', im0)[1].tobytes()
            dict_result=dict()
            dict_result["results"]=result

            cv2.imshow('current_img', im0) 

            # sock.send(json.dumps(dict_result))
           
            """  dict_result['img'] =base64.b64encode(im0).decode('utf-8') """
            
            
            """   response = sqs.send_message(
            QueueUrl=queue_url,
            DelaySeconds=10,
            MessageAttributes={
                'folder': {
                    'DataType': 'String',
                    'StringValue': p.stem 
                },
                'img': {
                    'DataType': 'Number',
                    'StringValue': str(seen)
                }
            },
            MessageBody=(
               str(dict_result)
            )
            ) """
            #yield json.dumps(dict_result)
        else:
            # im0 =cv2.resize(im0, (1280,720))
            PIL_image= Image.fromarray(im0.astype('uint8'), 'RGB')
            text = f"{s}"            
            mask_image = font.getmask(text[:-2], "L")
            img = Image.new("RGBA", mask_image.size, "white")
            img.im.paste(color, (0, 0) + mask_image.size, mask_image)  # need to use the inner `img.im.paste` due to `getmask` returning a core
            PIL_image.paste(img, (0, 0))
            open_cv_image = np.array(PIL_image) 
            im0 = cv2.imencode('.jpg', open_cv_image)[1].tobytes()
            yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + im0 + b'\r\n')
            

            

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    # if save_txt or save_img:
    #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
    #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)


@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route('/loading_image')
def get_image():    
    return send_file("loading.gif", mimetype='image/gif')


@app.route('/detect',methods = ['POST', 'GET'])
def video_feed():
    """Video streaming home page."""
    if not request.files.getlist('myfile'):
        opt.source, opt.save_txt = update_options(request)
    else:
        uploaded_file = request.files['myfile']
        url = 'test_public_123'+str(uploaded_file.filename)
        uploaded_file.save(url)
        opt.save_txt = None
        opt.source = url 

    return Response(detect(opt), mimetype='multipart/x-mixed-replace; boundary=frame')



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
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--port', default=5000, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

    # Just in case one dimension if provided, i.e., if 640 is provided, image inference will be over 640x640 images
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))

    # Check requirements are installed
    check_requirements(requirements=ROOT.parent / 'requirements.txt',exclude=('tensorboard', 'thop'))

    # Load model
    opt.device = select_device(opt.device)
    model = DetectMultiBackend(opt.weights, device=opt.device, dnn=opt.dnn, data=opt.data, fp16 = opt.half)
    stride, names, pt = model.stride, model.names, model.pt

    detect(opt)

    # Run app
    app.run(host="0.0.0.0", port=opt.port, debug=True) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)

   
