from PIL import Image
import numpy as np
import os


# img = Image.open("./img/图11.jpg")
# img_arr = np.asarray(img)
# np.set_printoptions(threshold=np.inf)
# print(img_arr)
# print(img_arr.shape)

# def testNp():
#     x = np.arange(8)
#     # x[0]=[1,2]
#
#     y=[1,2,3]
#     y[0] = [1,2]
#
#     z=x.tolist()
#     z[0] = [1,2]
#     print(x)
#     print(y)
#     print(z)
# testNp()

# def readFolder(folderSrc):
#     """
#     传入文件夹地址
#     返回所有文件
#     """
#     files = os.listdir(folderSrc)
#     return files
# folderSrc = ".\data\标准宋体\JPEGImages"
# files = readFolder(folderSrc)
# print(files)

# from PIL import Image
# img_path = '.\data\标准宋体\JPEGImages\GB1_R.jpg'
# img = Image.open(img_path)
# np.set_printoptions(threshold=np.inf)
# re_img = np.asarray(img)
# Image.fromarray(np.uint8(re_img)) # 转为图片时用到
# print(re_img)

# -*- coding:utf-8 -*-
class DataTest:
    def __init__(self, id, address):
        self.id = id
        self.address = address
        self.d = {self.id: 1,
                  self.address: "192.168.1.1"
                  }
        self.testList = [[11,22],[33,44]]

    def __getitem__(self, key):
        return self.testList[key]


data = DataTest(1, "192.168.2.11")
print(data[1][1])

# class Animal:
#     def __init__(self, animal_list):
#         self.animals_name = animal_list
#
#     def __getitem__(self, index):
#         return self.animals_name[index]
#
# animals = Animal(["dog","cat","fish"])
# for animal in animals:
#     print(animal)
