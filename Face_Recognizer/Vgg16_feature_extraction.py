#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 引入vgg16网络
from tensorflow_vgg import vgg16
from tensorflow_vgg import utils

# 引入mtcnn_findFace
from face_detect_main import mtcnn_findFace
import tensorflow as tf
from scipy import misc
import os
import cv2
import warnings
warnings.filterwarnings('ignore')


# In[2]:


graph = tf.Graph()
with graph.as_default():
    # 构建vgg16网络并加载参数
    vgg = vgg16.Vgg16()
    input_ = tf.keras.layers.Input(
        [224, 224,
         3])  # tf.compat.v1.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.name_scope("content_vgg"):
        vgg.build(input_)


def feature_extraction(imgs):
    path = "./timg.jpg"
    batch = []

    if len(imgs) > 0:
        if imgs.ndim > 3:
            for i in range(imgs.shape[0]):
                cv2.imwrite("./result_output/face_feature.png", imgs[i])
                # 调用开源代码实现的函数load_image加载人脸图像
                img = utils.load_image(path)
                batch.append(img)
        else:

            cv2.imwrite("./result_output/face_feature.png", imgs)
            img = utils.load_image(path)
            batch.append(img.reshape((1, 224, 224, 3)))
            batch = np.concatenate(batch)

        with tf.compat.v1.Session(graph=graph) as sess:
            feed_dict = {input_: batch}
            # 前向传播获得vgg.fc7(就是第二个全连接层)的输出作为人脸特征
            feature = sess.run(vgg.fc7, feed_dict=feed_dict)
            if imgs.ndim == 3:
                feature = np.reshape(feature, (1, 4096))
        # 返回人脸特征
        return feature


# In[ ]:


if __name__ == '__main__':
    image_path = "./timg.jpg"
    img = misc.imread(os.path.expanduser(image_path), mode='RGB')
    images, rects = mtcnn_findFace(img)
    feature_extraction(images)

