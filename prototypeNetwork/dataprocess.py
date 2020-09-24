# -*- coding:utf-8 -*-
"""
@Time: 2020/09/17 11:05
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 数据预处理：读入数据 数据增强 数据规整...
"""
import numpy as np
import os
import cv2
from scipy import ndimage
import multiprocess as mp
import zipfile
import matplotlib.pyplot as plt

import torch
import torchvision

def unzip_dataset(dataset_dir,prepared_dataset_dir):
    if not os.path.exists(prepared_dataset_dir):
        os.makedirs(prepared_dataset_dir)
    for root,_,files in os.walk(dataset_dir):
        for f in files:
            #解压文件
            zip_file=zipfile.ZipFile(os.path.join(root,f),'r')
            #将解压后的文件提取到
            zip_file.extractall(prepared_dataset_dir)
            zip_file.close()

def read_alphabets(alphabet_directory_path,alphabet_directory_name):
    '''
    :param alphabet_directory_path:
    :param alphabet_directory_name:
    :return:
    '''
    datax=[]
    datay=[]
    characters=os.listdir(alphabet_directory_path)
    for character in characters:
        images=os.listdir(os.path.join(alphabet_directory_path,character))
        for image in images:
            image=cv2.resize(cv2.imread(os.path.join(alphabet_directory_path,character,image)),(28,28))
            # 对图片执行不同程度的翻转
            rotated_90=ndimage.rotate(image,90)
            rotated_180=ndimage.rotate(image,180)
            rotated_270=ndimage.rotate(image,270)

            datax.extend((image,rotated_90,rotated_180,rotated_270))
            datay.extend((alphabet_directory_name+'_'+character+'_0',alphabet_directory_name+'_'+character+'_90',alphabet_directory_name+'_'+character+'_180',alphabet_directory_name+'_'+character+'_270'))
    return np.array(datax),np.array(datay)


def read_images(base_directory):
    '''
    :param base_directory:
    :return:
    '''
    datax=None
    datay=None
    pool=mp.Pool(mp.cpu_count())
    results=[pool.apply(read_alphabets,args=(base_directory+'/'+directory+'/',directory)) for directory in os.listdir(base_directory)]
    pool.close()
    for result in results:
        if datax is None:
            datax=result[0]
            datay=result[1]
        else:
            datax=np.vstack([datax,result[0]])
            #datay=np.vstack([datay,result[1]])
            datay=np.concatenate([datay,result[1]])
    return datax,datay

def create_sample(n_way,n_support,n_query,datax,datay):
    """
    :param n_way:
    :param n_support:
    :param n_query:
    :param datax:
    :param datay:
    :return:
    """
    sample=[]
    # 随机选取n_way个class
    k=np.random.choice(np.unique(datay),n_way,replace=False)
    for cls in k:
        # 筛选出该cls对应的样本数据
        datax_cls=datax[datay==cls]
        # 对筛选出的数据进行随机打乱
        perm=np.random.permutation(datax_cls)
        # 从数据中选择出n_support+n_query个样本
        sample_cls=perm[:(n_support+n_query)] #sample_cls shape:(n_support+n_query,28,28,3)
        sample.append(sample_cls)
    sample=np.array(sample)
    sample=torch.from_numpy(sample).float() #(n_way,n_support+n_query,28,28,3)
    sample=sample.permute(0,1,4,2,3) #(n_way,n_support+n_query,3,28,28)

    return ({'images':sample,
             'n_way':n_way,
             'n_support':n_support,
             'n_query':n_query,
             })

def display_sample(sample):
    """
    辅助函数 将样本可视化出来
    :param sample:
    :return:
    """
    # sample_4D:(n_way*(n_support+n_query),3,28,28)
    sample_4D=sample.view(sample.shape[0]*sample.shape[1],*sample.shape[2:])
    # make a grid
    out=torchvision.utils.make_grid(sample_4D,nrow=sample.shape[1])
    plt.figure(figsize=(16,7))
    plt.imshow(out.permute(1,2,0))
