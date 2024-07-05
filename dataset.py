import os
import math
import cv2
from torchvision import transforms as T
import numpy
from PIL import Image
import numpy as np
import torch


def read_font_images(font_dir, is_train=True):
    """读取所有font图像并标注。"""
    txt_fname = os.path.join(font_dir, '../train.txt' if is_train else '../val.txt')
    with open(txt_fname, 'r') as f:
        images = f.read().split()
    features, labels = [], []
    for i, fname in enumerate(images):
        feature = Image.open(os.path.join(
            font_dir, 'JPEGImages', f'{fname}.jpg')).convert("1")
        # features.append(np.array(T.Resize((288, 288))(feature.copy())).astype(float))
        features.append(np.array(feature.copy()).astype(float))
        feature.close()
        label = Image.open(os.path.join(
            font_dir, 'SegmentationClassAug', f'{fname}.png'))
        # labels.append(np.array(T.Resize((288, 288))(label.copy())))
        labels.append(np.array(label.copy()))
        label.close()
    return features, labels


# 字体数据集
class FontSegDataset(torch.utils.data.Dataset):
    def __init__(self, is_train, font_dir):
        self.features, self.labels = read_font_images(font_dir, is_train)

    def __getitem__(self, idx):
        p1 = 288 - self.labels[idx].shape[0]
        p2 = 288 - self.labels[idx].shape[1]
        if p1 > 0  and p2 > 0:
            label_pad = np.pad(self.labels[idx], ((p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)), 'constant', constant_values=0)
            feature_pad = np.pad(self.features[idx], ((p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)), 'constant',constant_values=1.0)
            return ((torch.from_numpy(feature_pad).float().reshape([1, 288, 288])),
                                    torch.from_numpy(label_pad).reshape([288, 288]).long())
        else:
            h, w = self.labels[idx].shape[:2]
            start_row, end_row = int(h * 0.15), int(h * 0.9)
            start_col, end_col = int(w * 0.15), int(w * 0.9)
            label_cut = self.labels[idx][start_row:end_row, start_col:end_col]
            feature_cut = self.features[idx][start_row:end_row, start_col:end_col]
            p1 = 288 - label_cut.shape[0]
            p2 = 288 - label_cut.shape[1]
            label_pad = np.pad(label_cut, ((p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)), 'constant',
                               constant_values=0)
            feature_pad = np.pad(feature_cut, ((p1 // 2, p1 - p1 // 2), (p2 // 2, p2 - p2 // 2)), 'constant',
                                 constant_values=1.0)
            return ((torch.from_numpy(feature_pad).float().reshape([1, 288, 288])),
                    torch.from_numpy(label_pad).reshape([288, 288]).long())

    def __len__(self):
        return len(self.features)


if __name__ == "__main__":
    TrainDataset = FontSegDataset(True, "data/方正美黑简体")
    # f, l = TrainDataset[2]
    # print(f.numpy().shape)
    for i in range(3349):
        f, l = TrainDataset[i]
        np.set_printoptions(threshold=1e6)  # 设置打印数量的阈值
        if f.numpy().shape != (1,288,288):
            print(i, f.numpy().shape)
    # print(TrainDataset.features[0])
    # print(l.numpy())
