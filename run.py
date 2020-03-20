#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 10:12:28 2019

@author: wanggongpeng
"""
import argparse
import json
import os
import time

import cv2
import dlib
import imutils
import numpy as np
##GPU显存配置
import tensorflow as tf
from PIL import Image
from apscheduler.schedulers.background import BackgroundScheduler
from imutils.video import FPS
from imutils.video import VideoStream

from config.defaults import cfg
# 导入相关工具包
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from yolo_change import YOLO

os.environ["CUDA_VISIBLE_DEVICES"] = cfg.GPU.DEVICEID

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = cfg.GPU.MEMORY  # 占用GPU90%的显存
session = tf.Session(config=config)
# # # 视频网络地址配置，这里配置网络摄像头路径

RTSP_URL = "rtsp://admin:admin1234@172.20.104.28:554/h264/ch1/main/av_stream"
device_id = RTSP_URL.split(r"/")[-2]  # 获取设备名编号
video_name = RTSP_URL.split(r"/")[-1]  # 获取视频地址

info_txt = cfg.DATA.INFOTXT
# 参数列表
ap = argparse.ArgumentParser()
args = vars(ap.parse_args())


# 定时任务,定时向kafka写入数据
def timedTask():
    write_json(total_down, total_up)


# 写json函数
# 真正要执行的函数，每秒钟往文件中写入结果
def write_json(total_down, total_up):
    # 获得当前时间时间戳
    now = int(time.time())
    # 转换为其他日期格式,如:"%Y-%m-%d %H:%M:%S"
    timeStruct = time.localtime(now)
    strTime = time.strftime("%Y%m%d%H%M%S", timeStruct)
    json_data = {
        "data": [
            {
                "device_id": device_id,
                "video_name": video_name,
                "captime": strTime,
                "total_down": total_down,
                "total_up": total_up
            }
        ]
    }
    with open("result.json", "a+") as w_json:
        json.dump(json_data, w_json)


def start_video():
    # 初始化视频流
    # if a video path was not supplied, grab a reference to the webcam
    if not cfg.DATA.INPUT:
        print("[INFO] starting video stream...")
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        # vs = cv2.VideoCapture(RTSP_URL)
        vs = cv2.VideoCapture(cfg.DATA.INPUT)

    print(cfg.DATA.INPUT)
    return vs


# 越线值
total_down = 0  # 出现人数
total_up = 0  # 过线人数

# 加载配置信息，如果FRAME.CATE=0代表通道，1表示闸机
if cfg.FRAME.CATE == 0:
    cfg.merge_from_file('./config/asile.yaml')
else:
    cfg.merge_from_file('./config/gate.yaml')
cfg.freeze()


def main_func(args, vs, yolo):
    scheduler = BackgroundScheduler()  # 初始化任务函数
    # 添加调度任务
    # 调度方法为 timedTask，触发器选择 interval(间隔性)，间隔时长为 1 秒
    scheduler.add_job(timedTask, 'interval', seconds=cfg.KAFKA.PUSHINTER)  # 间隔为1秒
    # 启动调度任务
    scheduler.start()
    writer = None  # 写入对象，如果要写入视频，将实例化该对象
    W = None
    H = None  # W,H是我们框架的大小、
    ct = CentroidTracker(maxDisappeared=cfg.CRT.MAXDISAPPEARED,
                         maxDistance=cfg.CRT.MAXDISTANCE)  # 质心追踪对象，连续40帧脱靶则注销#maxt=120
    trackers = []  # 用于存储dlib相关追踪器的列表
    trackableObjects = {}  # 映射id的字典
    totalFrames = 0  # 已处理帧总数
    global total_up, total_down  # 向下运动的人数
    fps = FPS().start()  # 用于基准测试的每秒帧数估算器
    inference_times = []
    # 已完成所有初始化，下面遍历传入的帧
    while True:

        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        st = time.time()
        frame = vs.read()
        frame = frame[1] if cfg.DATA.INPUT else frame
        # if we are viewing a video and we did not grab a frame then we
        # have reached the end of the video
        if cfg.DATA.INPUT is not None and frame is None:
            break

        # resize the frame to have a maximum width of 500 pixels (the
        # less data we have, the faster we can process it), then convert
        # the frame from BGR to RGB for dlib
        frame = imutils.resize(frame, width=cfg.FRAME.WIDTH)  # 调整框架的最大宽度为500像素，拥有的像素越少，处理速度越快
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        et = time.time()
        # if the frame dimensions are empty, set them
        if W is None or H is None:
            (H, W) = frame.shape[:2]

        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []  # 保存检测到或追踪到的对象

        if totalFrames % 2 == 0:

            if totalFrames % cfg.FRAME.SKIPFRAMES == 0:
                # set the status and initialize our new set of object trackers
                status = "Detecting"
                trackers_a = []  # 追踪对象的列表
                st = time.time()
                image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
                boxs, class_names = yolo.detect_image(image)
                et = time.time()
                print('detection take time : ', et - st)
                for box in boxs:
                    box = np.array(box)
                    (minx, miny, maxx, maxy) = box.astype("int")
                    cY = int((miny + maxy) / 2.0)
                    if cY > int(H * cfg.CRT.MINCY) and cY < int(H * cfg.CRT.MAXCY):
                        tracker = dlib.correlation_tracker()  # 实例化dlib相关性追踪器
                        rect = dlib.rectangle(minx, miny, maxx, maxy)  # 将对象的边界框坐标传给dlib.rectangle,结果存储在rect中
                        cv2.rectangle(frame, (minx, miny), (maxx, maxy), (2, 255, 0), 2)
                        rects.append((minx, miny, maxx, maxy))
                        # 开始追踪
                        tracker.start_track(rgb, rect)

                        # add the tracker to our list of trackers so we can
                        # utilize it during skip frames
                        trackers_a.append(tracker)

            else:
                st = time.time()
                # loop over the trackers
                for tracker in trackers_a:
                    # set the status of our system to be 'tracking' rather
                    # than 'waiting' or 'detecting'
                    status = "Tracking"

                    # update the tracker and grab the updated position
                    tracker.update(rgb)
                    pos = tracker.get_position()

                    # unpack the position object
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (2, 0, 255), 2)
                    rects.append((startX, startY, endX, endY))
                et = time.time()
                tt = et - st

            # 画一条水平的可视化线（行人必须交叉才能被追踪），并使用质心跟踪器更新对象质心
            # draw a horizontal line in the center of the frame -- once an
            # object crosses this line we will determine whether they were
            cv2.line(frame, (int(W * 0), int(H * cfg.FRAME.LINE)), (int(W * 1), int(H * cfg.FRAME.LINE)), (0, 255, 0),
                     2)  # 闸机测试

            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)
            # 在下一个步骤中，我们将回顾逻辑，该逻辑计算一个人是否在框架中向上或向下移动：
            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                # to determine direction
                else:
                    # the difference between the y-coordinate of the *current*
                    # centroid and the mean of *previous* centroids will tell
                    # us in which direction the object is moving (negative for
                    # 'up' and positive for 'down')
                    """
                    我们获取给定对象的所有先前质心位置的y坐标值。
                    然后，我们通过获取current-object当前质心位置与current-object所有先前质心位置的平均值之间的差来计算方向。
                    我们之所以这样做是为了确保我们的方向跟踪更加稳定。
                    如果我们仅存储该人的先前质心位置，则我们可能会错误地计算方向。
                    """
                    to.centroids.append(centroid)

                    # check to see if the object has been counted or not
                    if not to.counted:

                        # if the direction is negative (indicating the object
                        # is moving up) AND the centroid is above the center
                        # line, count the object
                        """
                        检查方向是否为负（指示对象正在向上移动）以及质心是否在中心线上方。
                        在这种情况下，我们增加 totalUp  。
                        """
                        if to.centroids[0][1] < int(H * cfg.FRAME.LINE) and centroid[1] > int(H * cfg.FRAME.LINE):
                            total_down += 1
                            to.counted = True
                            to.flag = 'DOWN'
                        elif to.centroids[0][1] > int(H * cfg.FRAME.LINE) and centroid[1] < int(H * cfg.FRAME.LINE):
                            total_up += 1
                            to.counted = True
                            to.flag = 'UP'
                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # 屏显
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                if to.counted:
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
                else:
                    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)
                    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("Up", total_up),
                ("Down", total_down),
                ("Status", status),
            ]
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>totalDown", total_down)
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>totalUp", total_up)
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            if cfg.DATA.OUTPUT is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(cfg.DATA.OUTPUT, fourcc, 30,
                                         (W, H), True)

            # 写入操作
            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # increment the total number of frames processed thus far and
        # then update the FPS counter
        end_time = time.time()
        inference_times.append(end_time - st)
        totalFrames += 1
        fps.update()
    # stop the timer and display FPS information
    try:
        inference_time = sum(inference_times) / len(inference_times)  # 计算FPS
        fps1 = 1.0 / inference_time  # FPS计算方式
        print("---------------------------------------------------------")
        print("FPS is ..............{}".format(fps1))
    except Exception as e:
        print(e)
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    print('totaldown people:', total_down)
    print('totalup people:', total_up)
    # 写测试信息
    with open(info_txt, 'w') as f:
        f.write("[INFO] elapsed time: " + str("{:.2f}".format(fps.elapsed())) + "\n")
        f.write("[INFO] approx. FPS: " + str("{:.2f}".format(fps.fps())) + "\n")
        f.write('totaldown people: ' + str(total_down) + "\n")
        f.write('totalup people: ' + str(total_up))
    # release the video capture
    vs.release()
    # check to see if we need to release the video writer pointer
    if writer is not None:
        writer.release()
    # close any open windows
    cv2.destroyAllWindows()


# 创建一个追踪对象类
if __name__ == '__main__':
    yolo = YOLO()
    vs = start_video()
    main_func(args, vs, yolo)
