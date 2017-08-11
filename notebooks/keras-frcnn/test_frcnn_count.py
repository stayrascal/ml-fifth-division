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

def format_img(img, C):
    img_min_side = float(C.im.size)
    (height, width, _) = img.shape

    if width <= height:
        f = img_min_side/width
        new_height = int(f * height)
        new_width = int(img_min_side)
    else:
        f = img_min_side/height
        new_width = int(f * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= C.img_channels_mean[0]
    img[:, :, 1] -= C.img_channels_mean[1]
    img[:, :, 2] -= C.img_channels_mean[2]
    img /= C.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

def accumulate(l):
    it = itertools.groupby(l, operator.itemgetter(0))
    for key, subiter in it:
        yield key, sum(item[1] for item in subiter)

def main():
    cleanup()
    sys.setrecursionlimit(40000)
    config_output_filename = 'config.pickle'

    with open(config_output_filename, 'r') as f_in:
        C = pickle.load(f_in)

    # turn off any data augmentation at test time
    C.use_horizontal_flips = False
    C.use_vertical_flips = False
    C.rot_90 = False
    class_mapping = C.class_mapping

    if 'bg' not in cladd_mapping:
        class_mapping['bg'] = len(class_mapping)

    class_mapping = {v : k for k, v in class_mapping.iteritems()}
    print(class_mapping)
    class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
    C.num_rois = num_rois

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
        input_shape_features = (1024, None, None)
    else:
        input_shape_img = (None, None, 3)
        input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(C.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # define the base network (resnet here, can be VGG, Inception, etc)
    shared_layers = nn.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier_only = Model([feature_map_input, roi_input], classifier)

    model_classifier = Model([feature_map_input, roi_input], classifier)

    model_rpn.load_weights(C.model_path, by_name=True)
    model_classifier.load_weights(C.model_path, by_name=True)

    model_rpn.compile(optimizer='sgd', loss='mse')

