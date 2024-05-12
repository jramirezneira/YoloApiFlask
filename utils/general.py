# Utils by Henry Navarro

"""
This module contains utility functions and constants for various purposes.

"""


import os
from pathlib import Path
import cv2
import json # Henry


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv8API root directory
RANK = int(os.getenv('RANK', -1))

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized
    
def update_options(request):
    """
    Args:
    - request: Flask request object
    
    Returns:
    - source: URL string
    - save_txt: Boolean indicating whether to save text or not
    """
    
    # GET parameters
    if request.method == 'GET':
        #all_args = request.args # TODO: get all parameters in one line
        source = request.args.get('url')
        save_txt = request.args.get('save_txt')

    
    # POST parameters
    elif request.method == 'POST':
        json_data = request.get_json() #Get the POSTed json
        json_data = json.dumps(json_data) # API receive a dictionary, so I have to do this to convert to string
        dict_data = json.loads(json_data) # Convert json to dictionary 
        source = dict_data['url']
        save_txt = dict_data.get('save_txt', None) 

    # else:     

    # request_split= request.split("&")
    # source = request_split[0]
    # s3_folder = request_split[1]
    # save_txt = False   
    
    return source, save_txt#, s3_folder
