## 初试PyTorch神经网络（mnist手写数字识别）

我们先来尝试一个简单的例子：MINIST手写数字识别

这个实例是我自己写的一个非常简单的入门案例，大家务必动手操作一下，后面我会提出相应的操作要求，大家按操作要求操作后截图保存，方便后期检查。

MNIST数据集是一组常见的图像，常用于测评和比较机器学习算法的性能。其中六万幅图像用于训练模型，另外一万幅用于测试模型。

![image style="zoom: 50%;"](https://user-images.githubusercontent.com/74011275/190901453-baf522fc-9375-43b1-af0b-9d15fdec5ce2.png)
![image](https://user-images.githubusercontent.com/74011275/190901484-28ec12c9-a155-4ae9-8b83-07b68da2f88c.png)

<img src="深度学习实践入门指南.assets/image-20220916142124277.png" alt="image-20220916142124277" style="zoom: 50%;" />

这些大小为28*28像素的单色图像没有颜色。每个像素是一个0~255的数值，表示该像素的明暗度。

MINIST数据集的下载地址：训练数据： https://pjreddie.com/media/files/mnist_train.csv。 测试数据： https://pjreddie.com/media/files/mnist_test.csv。

### 2.实践指南

代码地址：

1.按照第三部分搭建好pytorch环境

2.用pycharm右键打开此代码文件（pycharm专业版可以用csu邮箱申请），点击右下角interpreter settings

<img src="深度学习实践入门指南.assets/image-20220918013630865.png" alt="image-20220918013630865" style="zoom: 33%;" />



然后添加interpreter,add local interpreter

<img src="深度学习实践入门指南.assets/image-20220918013832814.png" alt="image-20220918013832814" style="zoom:33%;" />

选择安装pytorch环境的那个虚拟环境中的python解释器

3.在此虚拟环境中安装所需的其他依赖，pip install requirements.txt

(注意一定要在此虚拟环境中安装，具体过程不再赘说，自行查阅)

4.运行dataset.py查看数据集的图像样式，注意修改数据集地址

5.训练模型，通过argparse传参的方式设置数据集地址和训练轮数，选择不同的激活函数，损失函数和优化器运行train.py,第一次运行使用默认配置就可以

6.使用训练好的模型进行推断，运行inference.py,查看单个数字图像的推理结果，注意传参的时候要传入和训练此模型时使用的相同参数

7.对训练好的模型的性能进行评价，运行performance.py,同样注意传参的时候要传入和训练此模型时使用的相同参数，查看模型的数字识别准确率并进行记录

TIPS：pycharm中传参的方式

<img src="深度学习实践入门指南.assets/image-20220918123918454.png" alt="image-20220918123918454" style="zoom: 33%;" />



**作业要求：尝试使用不同激活函数，损失函数和优化器训练模型（均改变传入的参数即可，激活函数，损失函数和优化器我每个都给了两个选择）。使用控制变量法训练多个不同的模型，比较这些模型的性能好坏，对测试结果进行截图并做出对比表格，分析实验结果得出激活函数，优化器，损失函数的更优选择，并查阅资料分析为什么，写出文档说明。**

**注意：用训练好的模型去推断或者评价时参数args的配置要和训练模型时采用的参数一一对应***
