### 图像风格迁移

**使用预训练模型**

这部分代码中提供了7个预训练的模型：wave.ckpt-done、cubist.ckpt-done、denoised_starry.skpt-done、mosaic.ckpt-done、scream.ckpt-done、feathers. ckpt-done。回到源码目录chapter_7/，在其中新建一个model文件夹，然后把需要使用的模型文件复制到这个文件夹models/wave.ckpt-done。接下来运行下面的命令可以生成一张风格化图像了

以wave.ckpt-done的为例，新建一个models 文件
夹， 然后把wave.ckpt-done复制到这个文件夹下，运行命令：

```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

成功风格化的图像会被写到generated/res.jpg。
model_file 后面指定了与训练的模型的文件位置。如果没有把预训练模型保存为models/wave.ckpt-done，也可以自行替换为相应的文件位置。–image_file表示需要进行风格化的图像，在这里指定的是img目录下名为
test.jpg 的示例图像），也可以使用自己的图像进行尝试，同样只需要指定合适的文件位置即可。

准备工作：

- 在地址http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 下载VGG16模型，将下载到的压缩包解压后会得到一个vgg16.ckpt 文件。新建一个文件夹pretrained，并将vgg16.ckpt 复制到pretrained 文件夹中。最后的文件路径是pretrained/vgg16.ckpt

- 在地址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下载COCO数据集。将该数据集解压后会得到一个train2014 文件夹，其中应该含有大量jpg 格式的图片。在chapter_7中建立到这个文件夹的符号链接：
```
ln –s <到train2014 文件夹的路径> train2014
```

训练wave模型：
```
python train.py -c conf/wave.yml
```

打开TensorBoard：
```
tensorboard --logdir models/wave/
```

训练中保存的模型在文件夹models/wave/中。

