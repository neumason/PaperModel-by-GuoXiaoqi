import torch
from torch import nn
from models.unet import Unet
from dataset import FontSegDataset
import os

DATA_BASE_URL = "data/标准宋体"
BATCH_SIZE = 4
EPOCHS = 300
IS_USE_GPU = True
GPU_DEVICE = 0
LEARNING_RATE = 0.0001
MODEL_NAME = "unet-%s-%depochs.pt"%(DATA_BASE_URL.split("/")[1],EPOCHS)


if(os.path.exists("checkpoint") == False):
    os.makedirs("checkpoint")

if __name__ == '__main__':
    TrainDataset = FontSegDataset(True, DATA_BASE_URL)
    batch_size = BATCH_SIZE
    # 定义数据集迭代器
    train_iter = torch.utils.data.DataLoader(
        TrainDataset, batch_size, shuffle=True, drop_last=True)
    print("1.数据集加载成功")
    # 定义网络
    net = Unet(35)
    print("2.网络定义成功")
    if not IS_USE_GPU:
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        counter = 0   #计数器
        epochs = EPOCHS
        if os.path.exists(log_dir):
            checkpoint = torch.load(log_dir)
            net.load_state_dict(checkpoint['model'])
            optimiser.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X)
                loss = loss_function(Y_hat, Y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            torch.save(net, 'checkpoint/'+MODEL_NAME)
        print("训练结束")
    else:
        net = net.cuda(GPU_DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        counter = 0   #计数器
        epochs = EPOCHS
        # train
        print("3.开始训练")
        for epoch in range(epochs):
            print('training_epoch', epoch+1, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X.cuda(GPU_DEVICE))
                loss = loss_function(Y_hat, Y.cuda(GPU_DEVICE))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            torch.save(net, 'checkpoint/'+MODEL_NAME)
        print("训练结束")