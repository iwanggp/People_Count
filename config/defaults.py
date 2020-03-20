# -*- coding: utf-8 -*-
# @Time    : 2020/3/12
# @Author  : gpwang
# @File    : defaults.py
# @Software: PyCharm
from yacs.config import CfgNode as CN

# --------------------------------------------------------------------------
# config definition
# --------------------------------------------------------------------------
_C = CN()
# --------------------------------------------------------------------------
# config of GPU
# --------------------------------------------------------------------------
_C.GPU = CN()
_C.GPU.DEVICEID = '7'  # device id
_C.GPU.MEMORY = 0.03  # memory

# --------------------------------------------------------------------------
# config of model
# --------------------------------------------------------------------------
_C.MODEL = CN()
# Name of model
_C.MODEL.NAME = "YoloV3"
# the path of model
_C.MODEL.PATH = "./model_data/best_yolo.h5"
# anchors_path
_C.MODEL.ANCHORS_PATH = './model_data/yolo_anchors.txt'
# classes_path
_C.MODEL.CLASSES_PATH = "./model_data/brainwash_classes.txt"
# model score
_C.MODEL.SCORE = 0.4
# model iou
_C.MODEL.IOU = 0.45
# model input size
_C.MODEL.IMAGE_SIZE = (416, 416)
# gpu nums
_C.MODEL.GPU_NUM = 1
# --------------------------------------------------------------------------
# config of input and output
# --------------------------------------------------------------------------
_C.DATA = CN()
# input path
_C.DATA.INPUT = '/Users/gongpengwang/Documents/video-2.mp4'
# _C.DATA.INPUT = '/data/share/imageAlgorithm/dataset/dataset/20191209闸机录屏/video-1.mp4'
_C.DATA.OUTPUT = './output/01.mp4'
_C.DATA.INFOTXT = './data/01.txt'  # 暂时测试
# --------------------------------------------------------------------------
# config of centroidtracker
# --------------------------------------------------------------------------
_C.CRT = CN()
_C.CRT.MAXDISAPPEARED = 0
_C.CRT.MAXDISTANCE = 50
_C.CRT.MINCY = 0.1

_C.CRT.MAXCY = 0.6

# --------------------------------------------------------------------------
# config of frame
# --------------------------------------------------------------------------
_C.FRAME = CN()
_C.FRAME.WIDTH = 600
_C.FRAME.LINE = 0.35
_C.FRAME.SKIPFRAMES = 20
_C.FRAME.CATE = 1  # 0代表通道，1代表闸机

# --------------------------------------------------------------------------
# config of kafka
# --------------------------------------------------------------------------
_C.KAFKA = CN()
_C.KAFKA.BOOTSTRAPSERVERS = "172.16.100.100:9092"
_C.KAFKA.PUSHINTER = 1

cfg = _C
