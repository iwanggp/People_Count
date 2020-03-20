# People_Count
行人越线计数
## YOLOv3 论文出处
#### YOLOv3: An Incremental Improvement
[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf)   
[[Original Implementation]](https://github.com/pjreddie/darknet) 
#### Demo
请查看**video.mp4**测试视频。 
 
#### 人头检测模型下载

[百度云地址](https://pan.baidu.com/s/13qmA-UbVsYsG3B9UD2C-sQ)  提取:6o6d
#### 项目概述
* 用YOLOv3做人头检测   
* 用视频追踪和越线分析做行人越线计数  

## 安装清单
##### 环境
* keras >= 1.4.0
* TensorFlow >= 1.12.0
* dlib

## 相关参数配置
##### 请查考config文件夹内容
* 本工程采用的时**yacs**项目配置工具进行配置的，使用该工具非常的边界。为我们以后项目改动和调参提供了很大的便利。


##### 获取代码
```
git clone git@github.com:iwanggp/People_Count.git
cd People_Count
pip3 install -r piplist.txt --user
```
## 测试效果
##### 准备需要越线计数的数据
  
##### 开始测试
```
python run.py
```
可以从output文件夹获得结果.   

## FPS值及准确度
##### 测试结果
从行人过线的和出现的人数进行比对(私有视频和隐私问题，不适合上传) 
##### 测试结果
* 在1080Ti上进行测试，获得测试结果如下.    

| video.mp4	| 过线人数 | 出线人数 | 正确率  | FPS |
| ----- |:--------:|:----------:|:----------:|:--------------:|
| Paper | 87| 237   | 97%          | 45           |


## Credit
```
@article{yolov3,
	title={YOLOv3: An Incremental Improvement},
	author={Redmon, Joseph and Farhadi, Ali},
	journal = {arXiv},
	year={2018}
}
```

## 参考链接
* [darknet](https://github.com/pjreddie/darknet)
* [Trainning code](https://github.com/iwanggp/flag-detection)
* [YOLOv3](https://github.com/qqwweee/keras-yolo3)