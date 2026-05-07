# UI-Fire-Alarm-Forest-Fire-Detection-Smoke-Detection-System-Dataset-YOLO--Graduation-Project-
本项目通过yolov11/yolov10/yolov9/yolov8/yolov7/yolov5训练自己的数据集，并开发可视化界面。实现了一个火灾烟雾实时检测系统。可提供整套代码(含详细注释)、训练好的权重、数据集、测试视频和详细说明文档。可以部署到树莓派、香橙派、Jetson Nano、瑞芯微RK3588等开发板上，也可调用摄像头输入视频流进行实时推理。

# 一、前言
本项目通过yolov11/yolov10/yolov9/yolov8/yolov7/yolov5训练自己的数据集，并开发可视化界面。实现了一个火灾烟雾实时检测系统，操作视频和效果展示如下：
 [【yolov11/yolov10/yolov9/yolov8/yolov7/yolov5火灾烟雾检测系统-界面+视频实时检测+数据集(原创算法-毕业设计)】 https://www.bilibili.com/video/BV1FG41127H3/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a](https://www.bilibili.com/video/BV1FG41127H3/?share_source=copy_web&vd_source=138d2e7f294c3405b6ea31a67534ae1a)。
 
**可提供整套代码(含详细注释)、训练好的权重、数据集、测试视频和详细说明文档。可以部署到树莓派、香橙派、Jetson Nano、瑞芯微RK3588等开发板上，也可调用摄像头输入视频流进行实时推理。**
## 1、项目介绍
火灾是一种常见而严重的安全威胁，可以造成人员伤亡和财产损失。因此，提前检测并及时报警火灾发生对于保护人们的生命和财产具有重要意义。烟雾是火灾中最早出现的迹象之一，因此烟雾检测系统在火灾预警和灭火工作中起着关键作用。本设计旨在开发一个能够及时、准确地检测火灾烟雾的系统，其主要目标包括：实时监测环境中的烟雾并判定是否为火灾烟雾；提供可靠的火灾烟雾检测结果。
 我们的项目可为兄弟们的毕设、课设、大作业等提供参考，可训练自己的数据集，可以换成yolov11/yolov10/yolov9/yolov8/yolov7/yolov5各种版本的权重。包含特别详细的read.md文件和常见问题解答，关于本项目的任何问题都能在其中找到答案，对刚接触深度学习、目标检测的小白非常友好，兄弟们放心哈。
 ![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/d1acc2ad230f6b04cef738bece0b3e8a.png#pic_center)


## 2、图片测试效果展示
可以看到，我们实验室的项目能对图片、视频中出现的火灾、烟雾情况进行有效检测，且准确率较高。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/317149817b127ed954ac5f2e7fddd179.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/aaee57d4ba92766401392be4e1b49252.png)

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/52af163a855055fc483736d57a2f52b9.png)
# 二、项目环境配置
不熟悉pycharm的anaconda的大兄弟请先看这篇csdn博客，了解pycharm和anaconda的基本操作。
[https://blog.csdn.net/ECHOSON/article/details/117220445](https://blog.csdn.net/ECHOSON/article/details/117220445)
anaconda安装完成之后请切换到国内的源来提高下载速度 ，命令如下：

```python
conda config --remove-key channels
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.bfsu.edu.cn/anaconda/cloud/pytorch/
conda config --set show_channel_urls yes
pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
```
首先创建python3.8的虚拟环境，请在命令行中执行下列操作：

```python
conda create -n yolov8 python==3.8.5
conda activate yolov8
```
## 1、pytorch安装（gpu版本和cpu版本的安装)
实际测试情况是yolov10/yolov9/yolov8/yolov7/yolov5在CPU和GPU的情况下均可使用，不过在CPU的条件下训练那个速度会令人发指，所以有条件的小伙伴一定要安装GPU版本的Pytorch，没有条件的小伙伴最好是租服务器来使用。GPU版本安装的具体步骤可以参考这篇文章：[https://blog.csdn.net/ECHOSON/article/details/118420968](https://blog.csdn.net/ECHOSON/article/details/118420968)。
需要注意以下几点：
1、安装之前一定要先更新你的显卡驱动，去官网下载对应型号的驱动安装
2、30系显卡只能使用cuda11的版本
3、一定要创建虚拟环境，这样的话各个深度学习框架之间不发生冲突
我这里创建的是python3.8的环境，安装的Pytorch的版本是1.8.0，命令如下：

```python
conda install pytorch==1.8.0 torchvision torchaudio cudatoolkit=10.2 # 注意这条命令指定Pytorch的版本和cuda的版本
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cpuonly # CPU的小伙伴直接执行这条命令即可
```
安装完毕之后，我们来测试一下GPU是否可以有效调用：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a79b18bf74273e070fa9ff255a6fe311.png#pic_center)
## 2、pycocotools的安装

```python
pip install pycocotools-windows
```
## 3、其他包的安装
另外的话大家还需要安装程序其他所需的包，包括opencv，matplotlib这些包，不过这些包的安装比较简单，直接通过pip指令执行即可，我们cd到yolov8/yolov7/yolov5代码的目录下，直接执行下列指令即可完成包的安装。

```python
pip install -r requirements.txt
pip install pyqt5
pip install labelme
```
# 三、yolov8/yolov7/yolov5火灾烟雾检测系统
## 1、yolov8火灾烟雾检测算法
yolov8是yolo系列的最新算法，检测效果优于之前的所有的yolo算法。这里，我们采用了ultralytics官方版本的yolov8来检测火灾烟雾。

在学习Yolov8之前，我们需要对Yolov8所做的工作有一定的了解，这有助于我们后面去了解网络的细节，Yolov8在预测方式上与之前的Yolo并没有多大的差别，依然分为三个部分：分别是Backbone，FPN以及Yolo Head。

Backbone是Yolov8的主干特征提取网络，输入的图片首先会在主干网络里面进行特征提取，提取到的特征可以被称作特征层，是输入图片的特征集合。在主干部分，我们获取了三个特征层进行下一步网络的构建，这三个特征层我称它为有效特征层。

FPN是Yolov8的加强特征提取网络，在主干部分获得的三个有效特征层会在这一部分进行特征融合，特征融合的目的是结合不同尺度的特征信息。在FPN部分，已经获得的有效特征层被用于继续提取特征。在YoloV8里依然使用到了Panet的结构，我们不仅会对特征进行上采样实现特征融合，还会对特征再次进行下采样实现特征融合。

Yolo Head是Yolov8的分类器与回归器，通过Backbone和FPN，我们已经可以获得三个加强过的有效特征层。每一个特征层都有宽、高和通道数，此时我们可以将特征图看作一个又一个特征点的集合，每个特征点作为先验点，而不再存在先验框，每一个先验点都有通道数个特征。Yolo Head实际上所做的工作就是对特征点进行判断，判断特征点上的先验框是否有物体与其对应。Yolov8所用的解耦头是分开的，也就是分类和回归不在一个1X1卷积里实现。

因此，整个YoloV8网络所作的工作依然就是 特征提取-特征加强-预测先验框对应的物体情况。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/a0f18a577c0988b16d47303dcb5dc371.png#pic_center)


## 2、算法界面设计
这里我们使用了Streamlit来实现火灾烟雾检测系统界面的开发。Streamlit 是一个开源的 Python 库，用于创建交互式的数据科学和机器学习应用程序。它使得构建数据驱动的web应用变得简单易用，无需繁琐的前端开发，只需要使用 Python 进行开发即可。其实就是用Python语言写一个本地Web，应用场景常常是机器学习可视化（当然也可做其他的图表分析等），不需要任何Web前后端开发经验（事实上我也没有）。注意我们写出的Web只有本地可以访问，要接入互联网还需要其他进阶方法！我用下来整体体会是这个库集成度很高，功能都是完整地打包入完整的API里了，所以使用起来非常简单快捷。但集成度高也有缺点，例如设计的自由度较低，诸如前端说明字的位置、大小等不能直接实现，需要借助HTML、CSS样式表等进阶手段完成。

如图所示，在使用pycharm运行我们这个火灾烟雾检测系统时，首先需在pycharm右下角选中Pytorch环境下的python解释器（如新手小白没有pytorch环境，参考[http://t.csdn.cn/cI5BF](http://t.csdn.cn/cI5BF)进行配置即可），然后点击终端，点击下拉，选择Command Prompt进入pytorch环境下的执行终端。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/619e785f84ffa17942929b997c4d06e9.png#pic_center)
然后，在该终端中输入命令：

```python
streamlit run app.py
```
初次使用需要输入邮箱，如图所示，随便输一个即可，然后按回车键，就能打开我们项目的可视化界面进行操作了。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/eef2c0bac06216e01004f0ee584b0a92.png#pic_center)
另外，若pycharm出现如下版本提醒，可以不管，代码一样运行，没必要按它要求的版本重新装包，亲测有效。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/313df0fdf3916219ef612b164d1c0d23.png#pic_center)
运行界面展示如图，非常简洁易懂，容易上手。我们的项目支持图片、视频、实时摄像头三种检测模式，可以选择调用自己训练的权重或yolo官网给出的权重进行火灾检测。这里我们提供了一个在自己数据集上训练的权重供兄弟们调用：fire_detecter.pt，这个权重测出来的map达到93.6%，对于毕业设计或者课程设计完全够用。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/749bc89d1081d0fc9b876e327e8356de.png#pic_center)



# 四、火灾、烟雾检测自建数据集
## 1、数据集介绍
我们实验室手动收集、整理、标注了一个高质量的火灾、烟雾检测数据集，包含6940张火灾图片和对应的xml格式标签。我们对图片中的火焰（fire）、烟雾（smoke）使用软件进行了手动标注，并将其划分为训练集、验证与测试集。本数据集可直接用于训练yolo系列等神经网络，可提供给兄弟们的毕设、课设项目及企业课题进行使用。数据集展示如下：
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6f50eedbb27c031168c549ebc88a1f02.png#pic_center)

其中“annotations”包含了6940个手动标注的xml格式标签，“train_list”和“val_list”文件夹里面为训练集、验证集划分对应的txt，“images”文件夹里面为6940张JPG格式的包含火焰、烟雾的图片。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/ff2942bde69969fda08dd9cbde098835.png#pic_center)

# 五、训练曲线等介绍
我们的项目代码还能自动生成训练过程的loss损失曲线、map平均准确度曲线，不用手动画（太麻烦了，能用代码做的事尽量不手动），兄弟可以直接将这些图插入论文或课设报告中。当然，也可以自己训练，重新生成对应的图。训练结束后，这些图和训练数据会存放在根目录下的runs文件夹中。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/0e01b3e8d38b80358d9c0c115df26d21.png#pic_center)

包含完整word版本说明文档，可用于写论文、课设报告的参考。
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/6e000f122c9653e9496b03cb011e1ba9.png#pic_center)
# 六、全套资源获取(yolov11/yolov10/yolov9/yolov8/yolov7/yolov5版本均可提供)
**可提供整套代码、训练好的权重、数据集、还有测试视频和详细说明文档。代码有详细注释，包全程指导，任何问题都可以随时问我。不过有的时候我太忙，可能不会及时回复消息，看到了肯定回你哈**
```python
获取整套代码、测试视频、训练好的权重和说明文档(有偿)
上交硕士，技术够硬，也可以指导深度学习毕设、大作业等。
--------------->qq------------
           3582584734
------------------------------
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/c7e9309a04bd6f22ae3f1138149f65ea.png)


