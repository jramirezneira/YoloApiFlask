# Utils by Henry Navarro

"""
This module contains utility functions and constants for various purposes.

"""


import os
from pathlib import Path

import json # Henry


FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv8API root directory
RANK = int(os.getenv('RANK', -1))
    
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
