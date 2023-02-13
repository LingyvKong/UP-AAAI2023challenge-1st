1.数据集准备：
把官方data.zip中的图片解压到dataset/data/中；

**或者**把dataset/train_plus和dataset/val复制到你的data路径，并修改yaml中的数据集路径
这是我们基于官方图片生成的少量补充图片，详细描述见技术报告，你也可以通过以下方式自己生成：
- 修改脚本dataview.py中的路径并运行main_split()函数，划分训练集和验证集
- 运行脚本dataview.py中的main_train_plus()函数，生成train_plus补充训练集


2.训练：
将resnet_qavgg第二十行的deploy设置为False
运行如下指令进行训练：
python train.py --config configs/local_AAAI/qavgg.yaml

3.部署：
将resnet_qavgg第二十行的deploy = False改为True
运行如下指令转换onnx：
python toonnx.py --config configs/local_AAAI/qavgg.yaml --save_prefix 128_5conv --input_size 3x28x28

4.其他代码说明：
up/models/backbones/resnet_qavgg.py中包含了使用的backbone
up/models/backbones/qarepvgg_block.py 代码来源于https://github.com/Deci-AI/super-gradients/commit/22738cb04dcc6fcc9c3d01e635411a47f02e682c
up/tasks/cls/data/cls_transforms.py中增加了一部分数据增强
dataview.py可以用于统计数据集，切分数据集，数据扩增等功能
128_5conv.onnx为转换后的模型文件

The code is highly borrowed from: https://github.com/ModelTC/United-Perception

## United Perception

<img src=./up-logo.png width=50% />


## 文档

[up 官方文档](https://modeltc-up.readthedocs.io/en/latest/index.html)
