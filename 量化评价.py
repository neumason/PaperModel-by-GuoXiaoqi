from sqlite3 import DatabaseError
import torch
from PIL import Image
from torch import nn
from evaluate import fwiou, generate_matrix, miou , mpa
from models.mynet import mynet
from dataset1 import FontSegDataset
import matplotlib.pyplot as plt
import numpy as np


MODEL_PATH ="checkpoint/mynet-标准宋体-50epochs.pt"
DATA_BASE_URL = "data/标准宋体"
TestDataset = FontSegDataset(False, DATA_BASE_URL)
origin = TestDataset[3][1].numpy()
predict = np.array(Image.open(r'./LZY/第四章/output/标准宋体/predict/GB4_R.jpg').convert('L'))
# predict = np.array(Image.open(r'./LZY/第四章/output/标准宋体/JPEGImages/GB3_R.jpg').convert('L'))
if __name__ == '__main__':
    miousum = 0
    mpasum = 0
    fwiousum = 0
    # l = 100
    matrix =generate_matrix(origin,predict)
    mpasum += mpa.Pixel_Accuracy_Class(matrix)
    miousum += miou.Mean_Intersection_over_Union(matrix)
    fwiousum += fwiou.Frequency_Weighted_Intersection_over_Union(matrix)

        # if i and i % 100 == 0:
        #     print(i)
    print("mpa:%f"%(mpasum))
    print("miou:%f"%(miousum))
    print("fwiou:%f"%(fwiousum))