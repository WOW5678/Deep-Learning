# -*- coding:utf-8 -*-
"""
@Time: 2020/09/17 11:00
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 程序入口
"""
import numpy as np


import multiprocess as mp
import os
#import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm import tqdm_notebook
from tqdm import tnrange

# 加载其他模块的函数或者类
import dataprocess
import build_model

# 检查GPU是否可用
print(torch.cuda.is_available())

# train
def train(model,optimizer,train_x,train_y,n_way,n_support,n_query,max_epoch,epoch_size):
    '''

    :param model:
    :param optimizer:
    :param train_x:
    :param train_y:
    :param n_way:
    :param n_support:
    :param n_query:
    :param max_epoch:
    :param epoch_size:
    :return:
    '''
    model.train()

    scheduler=optim.lr_scheduler.StepLR(optimizer,1,gamma=0.5,last_epoch=-1)
    epoch=0
    stop=False

    while epoch <max_epoch and not stop:
        running_loss=0.0
        running_acc=0.0

        for episode in tnrange(epoch_size,desc='Epoch {:d} train'.format(epoch+1)):
            sample=dataprocess.create_sample(n_way,n_support,n_query,train_x,train_y)
            optimizer.zero_grad()

            loss,output=model.set_forward_loss(sample)
            running_loss+=output['loss']
            running_acc+=output['acc']

            loss.backward()
            optimizer.step()

        epoch_loss=running_loss/epoch_size
        epoch_acc=running_acc/epoch_size
        print('Epoch:{:d}--Loss:{:.4f}--ACC:{:.4f}'.format(epoch+1,epoch_loss,epoch_acc))
        epoch+=1
        scheduler.step()

# test
def test(model,test_x,test_y,n_way,n_support,n_query,test_episode):
    """
    :param model:
    :param test_x:
    :param test_y:
    :param n_way:
    :param n_support:
    :param n_query:
    :param test_episode:
    :return:
    """
    model.eval()
    running_loss=0.0
    running_acc=0.0
    for episode in tnrange(test_episode):
        sample=dataprocess.create_sample(n_way,n_support,n_query,test_x,test_y)
        loss,output=model.set_forward_loss(sample)
        running_loss+=output['loss']
        running_acc+=output['acc']
    avg_loss=running_loss/test_episode
    avg_acc=running_acc/test_episode
    print('Test results -- Loss:{:.4f}-- Acc:{:.4f}'.format(avg_loss,avg_acc))

if __name__ == '__main__':
    # step 1: load the dataset
    # 先对压缩的文件执行解压操作
    if not os.path.exists('data/Omniglot/images_background') or not os.path.exists('data/Omniglot/images_evaluation'):
        dataprocess.unzip_dataset('data/Omniglot_Raw','data/Omniglot')
    trainx,trainy=dataprocess.read_images('data/Omniglot/images_background')
    testx,testy=dataprocess.read_images('data/Omniglot/images_evaluation')
    print(trainx.shape,trainy.shape) #(77120, 28, 28, 3) (77120,)
    print(testx.shape,testy.shape)   #(52720, 28, 28, 3) (52720,)

    sample_example=dataprocess.create_sample(n_way=8,n_support=5,n_query=5,datax=trainx,datay=trainy)
    dataprocess.display_sample(sample_example['images'])

    # 创建模型
    model=build_model.load_protonet_conv(x_dim=(3,28,28),hid_dim=64,z_dim=64)
    optimizer=optim.Adam(model.parameters(),lr=0.001)

    n_way=60
    n_support=5
    n_query=5
    train_x=trainx
    train_y=trainy

    max_epoch=5
    epoch_size=20 # 一般都设置的比较大如2000 这里为了调试方便 设置较小
    train(model,optimizer,train_x,train_y,n_way,n_support,n_query,max_epoch,epoch_size)

    # 测试模型
    n_way=5
    n_support=5
    n_query=5

    test_x=testx
    test_y=testy

    test_episode=1000

    test(model,test_x,test_y,n_way,n_support,n_query,test_episode)

