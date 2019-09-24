本项目包含MTCNN（facenet实现）人脸检测方法，包含facenet和人脸特征提取方法。

项目中用到的一些模型需要下载放到本项目根目录下：

1. facenet的模型文件，请到['https://github.com/davidsandberg/facenet'] 地址去下载。

   modelname是20180402-114759，下载链接是：

   https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-

2. 从[Labeled Faces in the Wild网站[All images as gzipped tar file](http://vis-www.cs.umass.edu/lfw/lfw.tgz)來下载`lfw.tgz`。

3. 解压`lfw.tgz`到`face-recognition/data/`的目录下
4. 执行02-识别分类.ipynb来进行人脸特征获取（FaceNet），并训练人脸分类器，在“data”目录下产生“test”放置测试图像

