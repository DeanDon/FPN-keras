#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import skimage.io
from config import Config
import utils
import model_FPN as modellib
import visualize
from model import log
import tensorflow as tf
import xml.etree.ElementTree as ET
import scipy.sparse
import logging
import datetime
import keras.backend.tensorflow_backend as K

currentPath = os.path.split(os.path.realpath(__file__))[0]
curDate = datetime.date.today() - datetime.timedelta(days=0)
pid = os.getpid()
logName = '%s/log_%s_%s.log' % (currentPath, curDate, str(pid))
log_level = logging.DEBUG
logging.basicConfig(level=log_level,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S',
                    filename=logName,
                    filemode='w')

__CLASS__ = ['__backgroud', 'VOCcls']


def keras_bakend_config(gpuid):
    _gpu_options = tf.GPUOptions(allow_growth=False,
                                 per_process_gpu_memory_fraction=1.0,
                                 visible_device_list=gpuid)

    if not os.environ.get('OMP_NUM_THREADS'):
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=_gpu_options)
    else:
        num_thread = int(os.environ.get('OMP_NUM_THREADS'))
        config = tf.ConfigProto(intra_op_parallelism_threads=num_thread,
                                allow_soft_placement=True,
                                gpu_options=_gpu_options)
    _SESSION = tf.Session(config=config)
    K.set_session(_SESSION)


keras_bakend_config("0")
# %matplotlib inline

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to COCO trained weights
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")


class VOCConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "VOC"
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 2
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + n class

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 128

    LEARNING_RATE = 0.001
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 1000
    VOCDATA_PATH = '/path/to/VOCdevkit2007/VOC2007/'
    TRAIN_SET = os.path.join(VOCDATA_PATH, 'ImageSets/Main/trainval.txt')
    VAL_SET = os.path.join(VOCDATA_PATH, 'ImageSets/Main/val.txt')
    TEST_SET = os.path.join(VOCDATA_PATH, 'ImageSets/Main/test.txt')

    JPEG_IMAGES=os.path.join(VOCDATA_PATH, 'JPEGImages/')
    ANNOTATIONS=os.path.join(VOCDATA_PATH, 'Annotations/')
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1000


config = VOCConfig()
config.display()


class VOCDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_VOCdata(self, config, mode='train'):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.image_format = '.png'
        # self.all_samples=[line.strip() for line in  open(config.TRAIN_SET).readlines()]
        # self.num_sample=len(self.all_samples)
        if mode == 'train':
            list_image = [line.strip() for line in open(config.TRAIN_SET).readlines()]
            # list_image=list_image[0:len(list_image)/2]
        else:
            list_image = [line.strip() for line in open(config.TEST_SET).readlines()]
            # self.num_sample=len(self.all_samples)
            # list_image=self.all_samples[int(0.9*self.num_sample):]
        for index, _class_ in enumerate(__CLASS__[1:]):
            self.add_class("VOC", index + 1, _class_)
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        num_sample = len(list_image)
        logging.debug('Have train sample ' + str(num_sample))
        logging.debug('start get gt bbox from annotations')
        for i, info in enumerate(list_image):
            self.add_image("VOC", image_id=i, path=config.JPEG_IMAGES + info + self.image_format,
                           annotations_path=config.ANNOTATIONS + info + '.xml')
            if i % 1000 == 0 or i == (num_sample - 1):
                logging.debug('processing ' + str(i) + ' sample')

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "VOC":
            return info["VOC"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def _class_name_to_ind(self, class_name):
        """
        from class name get the class index
        :param class_name:
        :return class_index:
        """
        source_index = __CLASS__.index(class_name)
        return self.map_source_class_id('VOC.{}'.format(source_index))

    def load_gt(self, image_id):
        """
            Load image and bounding boxes info from XML file in the PASCAL VOC
            format.
            """
        annotation_path = self.image_info[image_id]['annotations_path']
        tree = ET.parse(annotation_path)
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text)
            y1 = float(bbox.find('ymin').text)
            x2 = float(bbox.find('xmax').text)
            y2 = float(bbox.find('ymax').text)
            cls = self._class_name_to_ind(obj.find('name').text.lower().strip())
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls

        return boxes, gt_classes


class InferenceConfig(VOCConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def read_annotations(annotation_path):
    tree = ET.parse(annotation_path)
    objs = tree.findall('object')
    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        boxes[ix, :] = [x1, y1, x2, y2]
        if x2-x1>=64 and y2-y1>=64:
            boxes[ix, :] = [x1+0.1*(x2-x1), y1+0.1*(y2-y1), x2-0.1*(x2-x1), y2-0.1*(y2-y1)]
        else:
            boxes[ix, :] = [x1, y1, x2, y2]
    return boxes

def test_vision():
    dataset_val = VOCDataset()
    dataset_val.load_VOCdata(config, 'val')
    dataset_val.prepare()

    inference_config = InferenceConfig()
    model = modellib.FPN(mode="inference", config=inference_config,
                         model_dir=MODEL_DIR)
    # Which weights to start with?
    init_with = "other"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', by_name=True)
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)
    for image_id in dataset_val.image_ids:
        image = dataset_val.load_image(image_id)
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=inference_config.IMAGE_MIN_DIM,
            max_dim=inference_config.IMAGE_MAX_DIM,
            padding=inference_config.IMAGE_PADDING)
        gt_bboxs, clss = dataset_val.load_gt(image_id)
        gt_bboxs = utils.resize_gt(gt_bboxs, scale, padding)
        # print "original_image", original_image
        # print "image_meta", image_meta
        # print "gt_bbox", gt_bboxs
        results = model.detect([image], verbose=1)
        for gt_bbox in gt_bboxs:
            cv2.rectangle(image, (int(gt_bbox[1]), int(gt_bbox[0])), (int(gt_bbox[3]), int(gt_bbox[2])),
                          color=(0, 0, 255),
                          thickness=2)
        for result in results:
            rois = result['rois']
            if rois.shape == (0, 4):
                cv2.imwrite('./' + str(image_id) + '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                print "no object"
                continue
            for bbox in rois:
                print bbox
                cv2.rectangle(image, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), color=(255, 0, 0),
                              thickness=2)
                cv2.imwrite('./' + str(image_id) + '.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    test_vision()
