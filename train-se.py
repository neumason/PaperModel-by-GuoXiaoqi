import torch
from torch import nn
from models.resnetse import ResNet18
from models.unet import Unet
from dataset import FontSegDataset
import os

DATA_BASE_URL = "data/dataset"
BATCH_SIZE = 4
EPOCHS = 300
IS_USE_GPU = True
GPU_DEVICE = 0
LEARNING_RATE = 0.0001
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True
CUDA_LAUNCH_BLOCKING=1
# MODEL_NAME = "resnet18-se-%s-%depochs.pt"%(DATA_BASE_URL.split("/")[1],EPOCHS)

# if(os.path.exists("checkpoint") == False):
#     os.makedirs("checkpoint")

if __name__ == '__main__':
    TrainDataset = FontSegDataset(True, DATA_BASE_URL)
    batch_size = BATCH_SIZE
    # 定义数据集迭代器
    train_iter = torch.utils.data.DataLoader(
        TrainDataset, batch_size, shuffle=True, drop_last=True)
    print("1.数据集加载成功")
    # 定义网络
    net = ResNet18(1,35)
    print("2.网络定义成功")
    if not IS_USE_GPU:
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        counter = 0   #计数器
        epochs = EPOCHS
        if os.path.exists("/home/cai/czw/stroke_segment/checkpoints/stroke2/ckpt_best_290.pth"):
            checkpoint = torch.load("/home/cai/czw/stroke_segment/checkpoints/stroke2/ckpt_best_290.pth")
            net.load_state_dict(checkpoint['model'])
            optimiser.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        # train
        print("3.开始训练")
        for epoch in range(start_epoch + 1, epochs + 1):
            print('training_epoch', epoch, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X)
                loss = loss_function(Y_hat, Y)
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            if epoch % 10 == 0:
                state = {'model': net.state_dict(), 'optimizer': optimiser.state_dict(), 'epoch': epoch}
                torch.save(state, "/home/cai/czw/code/checkpoints/stroke2/ckpt_best_%s.pth" % (str(epoch)))
            if epoch % 50 == 0:
                torch.save(net,
                           "/home/cai/czw/stroke_segment/checkpoints/" + "stroke2-%depochs.pt" % epoch)

        print("训练结束")
    else:
        net = net.cuda(GPU_DEVICE)
        loss_function = nn.CrossEntropyLoss()
        optimiser = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
        counter = 0   #计数器
        epochs = EPOCHS
        if os.path.exists("/home/cai/czw/stroke_segment/checkpoints/stroke7/ckpt_best_60.pth"):
            checkpoint = torch.load("/home/cai/czw/stroke_segment/checkpoints/stroke7/ckpt_best_280.pth")
            net.load_state_dict(checkpoint['model'])
            optimiser.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            print('加载 epoch {} 成功！'.format(start_epoch))
        else:
            start_epoch = 0
            print('无保存模型，将从头开始训练！')
        # train
        print("3.开始训练")
        for epoch in range(start_epoch + 1, epochs + 1):
            print('training_epoch', epoch, "of", epochs)
            for X, Y in train_iter:
                Y_hat = net(X.cuda(GPU_DEVICE))
                loss = loss_function(Y_hat, Y.cuda(GPU_DEVICE))
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                counter += 1
                if (counter % 100 == 0):
                    print("counter = ", counter, "loss = ", loss.item())
            if epoch % 10 == 0:
                state = {'model': net.state_dict(), 'optimizer': optimiser.state_dict(), 'epoch': epoch}
                torch.save(state, "/home/cai/czw/stroke_segment/checkpoints/stroke7/ckpt_best_%s.pth" % (str(epoch)))
            if epoch % 100 == 0:
                torch.save(net,
                           "/home/cai/czw/stroke_segment/checkpoints/" + "stroke7-%depochs.pt" % epoch)
        print("训练结束")