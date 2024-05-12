from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

import socket
import pickle
import struct 

HOST=''
PORT=8485

# s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
# print('Socket created')

# s.bind((HOST,PORT))
# print('Socket bind complete')
# s.listen(10)
# print('Socket now listening')

# conn,addr=s.accept()


model = YOLO("yolov8n.pt")


s="https://www.youtube.com/watch?v=MNn9qKG2UFI"
import pafy  # noqa
s = pafy.new(s).getbest(preftype='mp4').url

cap = cv2.VideoCapture(s)
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define region points
region_points = [(20, 400), (480, 404), (480, 360), (20, 360)]

# Video writer
# video_writer = cv2.VideoWriter("object_counting_output.avi",
#                        cv2.VideoWriter_fourcc(*'mp4v'),
#                        fps,
#                        (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=False,
                 reg_pts=region_points,
                 classes_names=model.names,
                 reg_counts=[(20, 400), (480, 404), (480, 360), (20, 360)],
                 draw_tracks=False)

cont=0
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    tracks = model.track(im0, persist=True, show=True)

    im0 = counter.start_counting(im0, tracks)
    # cv2.imshow("YOLOv8 Tracking", im0)
    # conn.send(im0)
    print ("send " +str(cont))
    cont=cont+1
    # video_writer.write(im0)

cap.release()
# video_writer.release()
cv2.destroyAllWindows()