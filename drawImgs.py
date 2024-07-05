from sqlite3 import DatabaseError
import os
import cv2
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import nn
from models.mynet import mynet
from models.unet import Unet
from models.segnet import SegNet
from dataset import FontSegDataset
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

MODEL_PATH = "checkpoints/stroke7-300epochs.pt"
DATA_BASE_URL = "data/方正兰亭黑"

font_colormap = [[255, 255, 255], [0, 0, 128], [0, 0, 64], [0, 128, 0], [0, 128, 128],
                 [0, 128, 64], [0, 192, 0], [0, 192, 128], [0, 64, 0], [0, 64, 128],
                 [128, 0, 0], [128, 0, 128], [128, 0, 64], [128, 128, 0], [128, 128, 128],
                 [128, 192, 0], [128, 192, 128], [128, 64, 0], [128, 64, 128], [192, 0, 0],
                 [192, 0, 128], [192, 128, 0], [192, 128, 128], [192, 192, 0], [192, 192, 128],
                 [192, 64, 0], [192, 64, 128], [64, 0, 0], [64, 0, 128], [64, 128, 0],
                 [64, 128, 128], [64, 192, 0], [64, 192, 128], [64, 64, 0], [64, 64, 128]]


def readFolder(folderSrc):
    """
    传入文件夹地址
    返回所有文件
    """
    files = os.listdir(folderSrc)
    filesLength = len(files)
    return files, filesLength


idx = 0


def pred(idx, files):
    net = torch.load(MODEL_PATH, map_location='cpu')
    TestDataset = FontSegDataset(False, DATA_BASE_URL)
    X = TestDataset[idx][0].unsqueeze(0)
    predict = net(X).argmax(dim=1).squeeze(0).numpy()
    # origin = TestDataset[idx][1].numpy()

    predict = predict.tolist()
    for i in range(len(predict)):
        for j in range(len(predict[i])):
            predict[i][j] = font_colormap[predict[i][j]]
    predict = np.array(predict)
    # np.set_printoptions(threshold=np.inf)

    # origin = origin.tolist()
    # print(origin)
    # for i in range(len(origin)):
    #     for j in range(len(origin[i])):
    #         origin[i][j] = font_colormap[origin[i][j]]
    # predict = np.array(origin)
    img = np.uint8(predict)
    # img = Image.fromarray(np.uint8(predict))
    cv2.imwrite('./result/stroke7/fzlth/{}'.format(files[idx]), img)
    # img.save('./output/标准宋体5/{}'.format(files[idx]))
    # img = Image.fromarray(np.uint8(predict), 'RGB')
    # img.save('./output/标准宋体/{}'.format(files[idx]))
    # plt.imshow(predict)
    # plt.savefig("./output/标准宋体")
    # plt.show()
    # plt.close()


if __name__ == '__main__':
    # folderSrc = ".\data\标准宋体\JPEGImages"
    # files, filesLength = readFolder(folderSrc)
    # filesSort = []
    # for idx in range(filesLength):
    #     filesSort.append(int(files[idx].split('.')[0].split('_')[0][2:]))
    # filesSort.sort()
    # files = []
    # for file in filesSort:
    #     files.append('GB'+str(file)+'_R.jpg')

    txt_fname = "./data/val.txt"
    with open(txt_fname, 'r') as f:
        filesSort = f.read().split()
    files = []
    for idx in filesSort:
        files.append(idx + '.jpg')

    for idx in range(200):
        pred(idx, files)
