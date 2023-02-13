"""
分析、处理数据集的脚本
author: Kong Lingyu
运行：
----------------1.数据统计分析----------------------------------------------
在数据集所在目录下直接运行 main_data_statistics()
实现的功能：
1. 把每个类别的第一张图片保存在./preview_data/文件夹中
2. 命令行输出每个类别的图片数，并绘制柱状图显示

----------------2. 数据增强效果可视化---------------------------------------
在数据集所在目录下直接运行 main_data_aug()

----------------3. 训练验证集划分和生成-------------------------------------
1. 命令行打印数据量top18的类别和对应的数据量
2. 生成符合testA分布的验证集val1.txt和对应的train1.txt
"""

import numpy as np
import pandas as pd
import os
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import copy

# meta_path = "./train.txt"
# train_path = "./train_plus.txt"
val_path = "dataset/val.txt"
# image_dir = ""
val_dir = "dataset/val/"
# train_plus_dir = "train_plus/"
preview_dir = "./preview_data/"
orig_img_path = './data/' + '33091.png'   # for data augment preview
class_num = 89

meta_path = "dataset/train.txt"
image_dir = "dataset/"
train_plus_dir = "dataset/train_plus/"
train_path = "dataset/train_plus.txt"


def make_dir(path, dele=False):
    if os.path.exists(path) and dele:
        shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        os.makedirs(path)


def main_data_statistics():
    make_dir(preview_dir)
    with open(meta_path, 'r') as f:
        imgs = [(image_dir+x).strip().split(" ") for x in f]
    imgs = np.array(imgs)
    cls_imgs = {}
    num = []
    for i in range(class_num):
        cls_imgs[i] = imgs[imgs[:,1]==str(i),0]
        num.append(len(cls_imgs[i]))
        print("class {} has {} images".format(i, len(cls_imgs[i])))
        shutil.copy(cls_imgs[i][0], preview_dir + str(i) + '-' + cls_imgs[i][0].split('/')[-1])

    # plot histograms of data nums for each category
    x = list(range(class_num))
    plt.figure(figsize=(16,4))
    plt.bar(x, num, width=0.8,color='#FF6347')
    for _x,_y in enumerate(num):
        plt.text(_x, _y+100, _y,ha='center',fontsize=5)
    plt.xticks(x, fontsize=8)
    plt.savefig('static.jpg', dpi=200)
    # plt.show()


def plot(imgs, with_orig=True, row_title=None, **imshow_kwargs):
    orig_img = Image.open(orig_img_path)
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [orig_img] + row if with_orig else row
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
    plt.tight_layout()
    plt.savefig("./aug.jpg")


def main_data_aug():
    plt.rcParams["savefig.bbox"] = 'tight'
    torch.manual_seed(0)
    orig_img = Image.open(orig_img_path)

    aug = T.RandomPerspective(distortion_scale=0.5, p=0.6)
    auged_imgs = [aug(orig_img) for _ in range(8)]
    plot(auged_imgs)

    # img = copy.copy(orig_img)
    # rotated_img = [TF.rotate(img, 90, expand=True)]
    # rotated_img[0].save("./rotimg.png")
    # plot(rotated_img)


def main_split():
    thresh_hold = 0.1   # 抽取的验证集小于原数据量的1%，则直接抽取；否则，抽取+生成
    np.random.seed(0)
    torch.manual_seed(0)
    with open(meta_path, 'r') as f:
        imgs = [(image_dir+x).strip().split(" ") for x in f]
    imgs = np.array(imgs)
    cls_imgs = {}
    num = []
    for i in range(class_num):
        cls_imgs[i] = imgs[imgs[:,1]==str(i),:]
        num.append(len(cls_imgs[i]))

    # print top 18 cls
    topk = 18
    top_num = pd.Series(num).sort_values(ascending = False)  # 从大到小
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print("topk cls and num:\n", top_num[:topk])
    flag_num = ([160]*18) + ([100]*(89 - 18))  # 前18类有160个作验证集，其余有100个
    i = 0

    # create validation set
    make_dir(val_dir, dele=True)
    train1 = []
    val1 = []
    aug_fun = T.Compose([T.RandomRotation(10),
                        T.RandomPerspective(distortion_scale=0.2, p=0.3),
                        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                        T.RandomAdjustSharpness(sharpness_factor=0, p=0.3),
                        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.4)])
    for cls_num in top_num.items():
        np.random.shuffle(cls_imgs[cls_num[0]])
        if cls_num[1] * thresh_hold >= flag_num[i]:
            for _i in range(flag_num[i]):
                val1.append(' '.join([str(cls_imgs[cls_num[0]][_i,0]), str(cls_imgs[cls_num[0]][_i,1])]) + '\n')
            for _i in range(flag_num[i], cls_num[1]):
                train1.append(' '.join([str(cls_imgs[cls_num[0]][_i,0]), str(cls_imgs[cls_num[0]][_i,1])]) + '\n')
        else:
            temp_n = int(cls_num[1] * thresh_hold) + 1
            gen_perimg = int((flag_num[i] - temp_n)/temp_n)
            for _i in range(temp_n):
                val1.append(' '.join([str(cls_imgs[cls_num[0]][_i,0]), str(cls_imgs[cls_num[0]][_i,1])]) + '\n')
                ori_img = Image.open(image_dir + cls_imgs[cls_num[0]][_i,0])
                for _j in range(gen_perimg):
                    aug_img = aug_fun(ori_img)
                    aug_img_path = val_dir + '{}_'.format(_j) + cls_imgs[cls_num[0]][_i,0].split("/")[-1]
                    aug_img.save(aug_img_path)
                    val1.append(' '.join([aug_img_path, str(cls_imgs[cls_num[0]][_i,1])]) + '\n')

            for _i in range(temp_n, cls_num[1]):
                train1.append(' '.join([str(cls_imgs[cls_num[0]][_i,0]), str(cls_imgs[cls_num[0]][_i,1])]) + '\n')

        i = i + 1

    with open(train_path,'w') as ft:
        ft.writelines(train1)
    with open(val_path,'w') as fv:
        fv.writelines(val1)

def main_modify_path():
    # from data/1234.png to data/data/1234.png
    with open(train_path, 'r') as f:
        imgs_t = ['data/' + x for x in f]
    with open('train1.txt','w') as f:
        f.writelines(imgs_t)
    with open(val_path, 'r') as f:
        imgs_v = [x for x in f]
    imgs_v2 = []
    for s in imgs_v:
        if s[0]=='d':
            imgs_v2.append('data/' + s)
        else:
            imgs_v2.append('val/' + s)
    with open('val1.txt','w') as f:
        f.writelines(imgs_v2)

def main_error_preview():
    # error_dict = {}
    # num = error_dict.values()
    num = [6,3,2,1,2,0,3,11,3,8,1,6,0,8,8,0,0,0,7,2,0,0,1,1,6,1,
    3,19,2,19,10,25,14,17,3,10,5,9,3,20,1,6,2,2,1,
    5,1,1,4,8,10,2,5,11,0,3,19,0,4,0,5,2,0,3,5,7,11,6,3,1,
    40,7,11,14,14,35,77,17,13,17,29,99,28,30,22,49,16,4,16]
    x = list(range(class_num))
    plt.figure(figsize=(16,4))
    plt.bar(x, num, width=0.8,color='#7B68EE')
    for _x,_y in enumerate(num):
        plt.text(_x, _y+10, _y,ha='center',fontsize=5)
    plt.xticks(x, fontsize=8)
    plt.savefig('error.jpg', dpi=200)

def main_train_plus():
    flag_num = 100
    np.random.seed(0)
    # 读取每个类的图片及数量
    with open(meta_path, 'r') as f:
        imgs = [(image_dir+x).strip().split(" ") for x in f]
    imgs = np.array(imgs)
    cls_imgs = {}
    num = []
    for i in range(class_num):
        cls_imgs[i] = imgs[imgs[:,1]==str(i),:]
        num.append(len(cls_imgs[i]))
    aug_fun = T.Compose([T.RandomRotation(10),
                        T.RandomPerspective(distortion_scale=0.2, p=0.3),
                        T.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
                        T.RandomAdjustSharpness(sharpness_factor=0, p=0.3),
                        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)])
    make_dir(train_plus_dir, dele=True)
    train_plus = []
    # 低于 flag_num 的使用数据增强随机生成补足
    for i in range(class_num):
        if num[i] < flag_num:
            gen_src = np.random.randint(num[i], size=flag_num-num[i])
            _j = 0
            for _i in gen_src:
                ori_img = Image.open(image_dir + cls_imgs[i][_i,0])
                aug_img = aug_fun(ori_img)
                aug_img_path = train_plus_dir + train_plus_dir + '{}_'.format(_j) + cls_imgs[i][_i,0].split("/")[-1]
                aug_img.save(aug_img_path)
                train_plus.append(' '.join([aug_img_path, str(cls_imgs[i][_i,1])]) + '\n')
                _j = _j + 1
    with open(train_path,'w') as ft:
        ft.writelines(train_plus)



if __name__=="__main__":
    # main_data_statistics()
    # main_data_aug()
    # main_split()
    # main_modify_path()
    # main_error_preview()
    main_train_plus()
