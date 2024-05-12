# import required libraries
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2

import json

# def detectaDuplicados(l):
#     print([item for item in set(l) if l.count(item) > 1])


# detectaDuplicados([3,1,1,3])
# detectaDuplicados([1,3,2,2])



# define and open video source
# stream = CamGear(source="rtsp://192.168.1.159:554/11", logging=True).start()

# # define required FFmpeg parameters for your writer
# output_params = {
#     "-clones": ["-f", "lavfi", "-i", "anullsrc"],
#     "-vcodec": "libx264",
#     "-preset": "medium",
#     "-b:v": "4500k",
#     "-bufsize": "512k",
#     "-pix_fmt": "yuv420p",
#     "-f": "flv",
# }

# # output_params = {
# #     "-acodec": "aac",
# #     "-ar": 44100,
# #     "-b:a": 712000,
# #     "-vcodec": "libx264",
# #     "-preset": "medium",
# #     "-b:v": "4500k",
# #     "-bufsize": "512k",
# #     "-pix_fmt": "yuv420p",
# #     "-f": "flv",
# # }




# # [WARNING] Change your YouTube-Live Stream Key here:
# YOUTUBE_STREAM_KEY = "phbq-55ve-tah4-h6jk-a29q"


# # Define writer with defined parameters
# writer = WriteGear(
#     output="rtmp://a.rtmp.youtube.com/live2/{}".format(YOUTUBE_STREAM_KEY),
#     logging=True,
#     **output_params
# )

# # loop over
# while True:

#     # read frames from stream
#     frame = stream.read()

#     # check for frame if Nonetype
#     if frame is None:
#         break

#     # {do something with the frame here}

#     # write frame to writer
#     writer.write(frame)

# # safely close video stream
# stream.stop()

# # safely close writer
# writer.close()