import os

from dataset import FontSegDataset

# -*- coding: utf-8 -*-

# 导入包
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from PIL import Image

"""
1.按照笔画类别分类
2.
"""
DATA_BASE_URL = "data/标准宋体"


def glyphClass():
    TrainDataset = FontSegDataset(False, DATA_BASE_URL)
    print(len(TrainDataset))
    res = set()
    for i in range(len(TrainDataset)):
        # if i % 100 == 0:
        # print(i)
        # print(res)
        # print(len(res))
        _, l = TrainDataset[i]
        l = l.numpy()
        for x in range(288):
            for y in range(288):
                if l[x][y] == 1:
                    res.add(i)
    print(res)
    print(len(res))


def readFolder(folderSrc):
    """
    传入文件夹地址
    返回所有文件
    """
    files = os.listdir(folderSrc)
    filesLength = len(files)
    return files, filesLength


def createFolder(folderSrc):
    os.makedirs(folderSrc, exist_ok=True)


def readAug(files):
    path = "data/标准宋体/SegmentationClassAug"
    # 读取图片，并转为数组
    # im = np.array(Image.open(path + '/GB1_R.png').convert('L'))
    # np.set_printoptions(threshold=np.inf)
    # print(len(im))
    # 打印数组
    for i in range(34, 35):
        for x in files:
            print(x)
            im = np.array(Image.open(path + '/' + x).convert('L'))
            for m in range(len(im)):
                for n in range(len(im[0])):
                    if im[m][n] == i:
                        im[m][n] = 255
            img = Image.fromarray(np.uint8(im), 'L')
            img.save('./output/标准宋体Aug/'+str(i)+'/{}'.format(x))


if __name__ == "__main__":
    # glyphClass()
    path = "data/标准宋体/SegmentationClassAug"
    # for i in range(13, 34):
    #     createFolder('./output/标准宋体Aug/' + str(i))
    files, fileslen = readFolder(path)
    readAug(files)
