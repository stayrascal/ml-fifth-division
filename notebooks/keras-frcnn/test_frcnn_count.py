import os
import cv2
import numpy as np
import sys
import pickle
import optparse import OptionParser
import time
import itertools
import operator
from keras_frcnn import config
from keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import datetime
import re
import subprocess

parse = OptionParser()
parse.add_option("-i", "--input_file", dest="input_file", help="Path to input video file.")
parse.add_option("-o", "--output_file", dest="output_file", help="Path to output video file.", default="output.mp4")
parse.add_option("-d", "--input_dir", dest="input_dir", help="Path to input working directory.", default="/home/zpwu/input")
parse.add_option("-u", "--output_dir", dest="output_dir", help="Path to output working directory.", default="/home/zpwu/output")
parse.add_option("-r", "--frame_rate", dest="frame_rate", help="Frame rate of the output video.", default=25.0)

(options, args) = parser.parse_args()
if not options.input_file:
    parser.errors('Error: path to vedio input_file must be specified. Pass --input-file to command line')

input_video_file = options.input_file
output_video_file = options.output_file
img_path = os.path.join(options.input_dir, '')
output_path = os.path.join(options.output_dir, '')
num_rois = 32
frame_rate = float(options.frame_rate)

def cleanup():
    print("cleaning up...")
    os.popen('rm -f ' + img_path + '*')
    os.popen('rm -f ' + output_path + '*')

def get_file_names(search_path):
    for (dirpath, _, filenames) in os.walk(search_path):
        for filename in filenames:
            yield filename
def convert_to_images():
    cam = cv2.VideoCapture(input_video_file)
    counter = 0
    while True:
        flag, frame = cam.read()
        if flag:
            cv2.imwrite(os.path.join(img_path, str(counter) + '.jpg'), frame)
            counter = counter + 1
        else:
            break
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def save_to_video():
    list_files = sorted(get_filenames(output_path), key= lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
    img0 = cv2.imread(os.path.join(output_path, '0.jpg'))
    height, width, layers = img0.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    videowriter = cv2.VideoWriter(output_video_file, fourcc, frame_rate, (width, height))
    for f in list_files:
        print("saving..." + f)
        img = cv2.imread(os.path.join(output_path, f))
        videowriter.writer(img)
    videowriter.release()
    cv2.destroyAllWindows()


