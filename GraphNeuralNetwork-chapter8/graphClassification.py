# -*- coding:utf-8 -*-
"""
@Time: 2020/09/28 10:12
@Author: Shanshan Wang
@Version: Python 3.7
@Function: 图分类
"""
import os
import urllib
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import numpy as np
import scipy.sparse  as sp
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import torch_scatter
from collections import Counter

# 功能函数定义
def tensor_from_numpy(x,device):
    return torch.from_numpy(x).to(device)

def normalization(adjacency):

    '''
    计算 L=D^-0.5 * (A+I) * D^-0.5,
    :param adjacency:sp.scr_matrix
    :return: 归一化后的邻接矩阵，类型为 torch.sparse.FloatTensor
    '''
    adjacency+=sp.eye(adjacency.shape[0]) # 增加自连接
    degree=np.array(adjacency.sum(1)) #
    d_hat=sp.diags(np.power(degree,-0.5).flatten()) #(334925,1)
    L=d_hat.dot(adjacency).dot(d_hat).tocoo() #coo_matrix (334925,334925)
    # 转换为torch.sparse.FloatTensor
    row=L.row.astype(np.int32) #ndarray [2021017]
    col=L.col.astype(np.int32) #ndarray [2021017]
    indices = torch.from_numpy(np.asarray([row,col])).long() #（2,2021017）
    #indices = torch.from_numpy(np.asarray([L.row, L.col])).long()
    values=torch.from_numpy(L.data.astype(np.float32))
    tensor_adjacency=torch.sparse.FloatTensor(indices,values,L.shape) #（334925,334925）
    return tensor_adjacency

## D&D数据
class DDDataset(object):
    url = "https://ls11-www.cs.tu-dortmund.de/people/morris/graphkerneldatasets/DD.zip"
    def __init__(self,data_root='data',train_size=0.8):
        self.data_root=data_root
        self.maybe_download()
        sparse_adjacency,node_labels,graph_indicator,graph_labels=self.read_data()
        self.sparse_adjacency = sparse_adjacency.tocsr() # csr_matrix
        self.node_labels = node_labels
        self.graph_indicator = graph_indicator # 这个是什么？不太理解
        self.graph_labels = graph_labels
        self.train_index, self.test_index = self.split_data(train_size)
        self.train_label = graph_labels[self.train_index]
        self.test_label = graph_labels[self.test_index]

    def split_data(self,train_size):
        unique_indicator = np.asarray(list(set(self.graph_indicator))) #(1178,)
        train_index, test_index = train_test_split(unique_indicator,
                                                   train_size=train_size,
                                                   random_state=1234)
        return train_index, test_index #(942,) (236,)

    def __getitem__(self, index):
        mask = self.graph_indicator == index
        node_labels = self.node_labels[mask]
        graph_indicator = self.graph_indicator[mask]
        graph_labels = self.graph_labels[index]
        adjacency = self.sparse_adjacency[mask, :][:, mask]
        return adjacency, node_labels, graph_indicator, graph_labels

    def __len__(self):
        return len(self.graph_labels)

    def read_data(self):
        data_dir=os.path.join(self.data_root,'DD')
        print('Loading DD_A.txt')
        adjacency_list = np.genfromtxt(os.path.join(data_dir, "DD_A.txt"),
                                       dtype=np.int64, delimiter=',') - 1 # ndarray(1686092, 2)
        print("Loading DD_node_labels.txt")
        node_labels = np.genfromtxt(os.path.join(data_dir, "DD_node_labels.txt"),
                                    dtype=np.int64) - 1 # ndarray （334925,）
        print("Loading DD_graph_indicator.txt")
        graph_indicator = np.genfromtxt(os.path.join(data_dir, "DD_graph_indicator.txt"),
                                        dtype=np.int64) - 1 # ndarray (334925,)
        print("Loading DD_graph_labels.txt")
        graph_labels = np.genfromtxt(os.path.join(data_dir, "DD_graph_labels.txt"),
                                     dtype=np.int64) - 1 # ndarray (1178,)
        num_nodes = len(node_labels) #334925

        sparse_adjacency = sp.coo_matrix((np.ones(len(adjacency_list)),
                                          (adjacency_list[:, 0], adjacency_list[:, 1])),
                                         shape=(num_nodes, num_nodes), dtype=np.float32) #coo_matrix (334925,334925)
        print("Number of nodes: ", num_nodes) # 334925
        return sparse_adjacency, node_labels, graph_indicator, graph_labels

    def maybe_download(self):
        save_path=os.path.join(self.data_root)
        if not os.path.exists(save_path):
            self.download_data(self.url,save_path)
        if not os.path.exists(os.path.join(self.data_root,'DD')):
            zipfilename=os.path.join(self.data_root,'DD.zip')
            with ZipFile(zipfilename,'r') as zipobj:
                zipobj.extractall(os.path.join(self.data_root))
                print('Extracting data from {}'.format(zipfilename))


    @staticmethod
    def download_data(url,save_path):
        '''
        数据下载工具，当原始数据不存在时会进行下载
        :param url:
        :param save_path:
        :return:
        '''
        print('downloading data from {}'.format(url))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data=urllib.request.urlopen(url)
        filename='DD.zip'
        with open(os.path.join(save_path,filename),'wb') as f:
            f.write(data.read())
        return True

# 模型定义
class GraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,use_bias=True):
        '''
        图卷积：L*X*\theta
        :param input_dim: 节点输入特征的维度
        :param output_dim: 输出特征的维度
        :param use_bias: 是否使用偏置
        '''
        super(GraphConvolution, self).__init__()
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.use_bias=use_bias
        self.weight=nn.Parameter(torch.Tensor(input_dim,output_dim))
        if self.use_bias:
            self.bias=nn.Parameter(torch.Tensor(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self,adjacency,input_feature):
        '''
        邻接矩阵是稀疏矩阵 因此在计算时使用稀疏矩阵乘法
        :param adjacency: (334925,334925)
        :param input_feature: (334925,89)
        :return: tensor (334925,32)
        '''
        support=torch.mm(input_feature,self.weight) #(334925,32)
        output=torch.sparse.mm(adjacency,support) #(334925,32)
        if self.use_bias:
            output+=self.bias
        return output

    def __repr(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_dim) + ' -> ' \
               + str(self.output_dim) + ')'

# Readout实现
def global_max_pool(x,graph_indicator):
    num=graph_indicator.max().item()+1
    return torch_scatter.scatter_max(x,graph_indicator,dim=0,dim_size=num)[0]

def global_avg_pool(x,graph_indicator):
    num=graph_indicator.max().item()+1 # 1178
    return torch_scatter.scatter_mean(x,graph_indicator,dim=0,dim_size=num)

# 基于自注意力机制的池化层
def top_rank(attention_score,graph_indicator,keep_ratio):
    '''
    基于给定的attention_score, 对每个图进行pooling操作.
    为了直观体现pooling过程，我们将每个图单独进行池化，最后再将它们级联起来进行下一步计算

    :param attention_score:使用GCN计算出的注意力分数，Z = GCN(A, X)
    :param graph_indicator:指示每个节点属于哪个图
    :param keep_ratio:要保留的节点比例，保留的节点数量为int(N * keep_ratio)
    :return:
    '''
    # TODO: 确认是否是有序的, 必须是有序的
    graph_id_list=list(set(graph_indicator.cpu().numpy()))
    mask=attention_score.new_empty((0,),dtype=torch.bool)
    for graph_id in graph_id_list:
        graph_attn_score=attention_score[graph_indicator==graph_id]
        graph_node_num=len(graph_attn_score) #[327]
        graph_mask=attention_score.new_zeros((graph_node_num,),dtype=torch.bool) #[327]
        keep_graph_node_num=int(keep_ratio*graph_node_num) #163
        _,sorted_index=graph_attn_score.sort(descending=True)
        graph_mask[sorted_index[:keep_graph_node_num]]=True
        mask=torch.cat((mask,graph_mask)) #(327)
    return mask

def filter_adjacency(adjacency,mask):
    '''
    根据掩码mask对图结构进行更新
    :param adjacency:torch.sparse.FloatTensor, 池化之前的邻接矩阵 （334925,334925）
    :param mask:torch.Tensor(dtype=torch.bool), 节点掩码向量 （334925，）
    :return:  torch.sparse.FloatTensor, 池化之后归一化邻接矩阵
    '''
    device=adjacency.device
    mask=mask.cpu().numpy()
    indices=adjacency.coalesce().indices().cpu().numpy() #(2,2021017)
    num_nodes=adjacency.size(0)#334925
    row,col=indices
    maskout_self_loop=row!=col #(2021017,)
    row = row[maskout_self_loop] #(1686092)
    col = col[maskout_self_loop] #(1686092)
    sparse_adjacency = sp.csr_matrix((np.ones(len(row)), (row, col)),
                                     shape=(num_nodes, num_nodes), dtype=np.float32)
    filtered_adjacency = sparse_adjacency[mask, :][:, mask] #(167169,167169)
    return normalization(filtered_adjacency).to(device)


class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim, keep_ratio, activation=torch.tanh):
        super(SelfAttentionPooling, self).__init__()
        self.input_dim = input_dim #96
        self.keep_ratio = keep_ratio #0.5
        self.activation = activation
        self.attn_gcn = GraphConvolution(input_dim, 1)

    def forward(self, adjacency, input_feature, graph_indicator):
        '''

        :param adjacency: Tensor(334925,334925)
        :param input_feature: (334925,96)
        :param graph_indicator: (334925)
        :return:
        '''
        attn_score = self.attn_gcn(adjacency, input_feature).squeeze() #(334925)
        attn_score = self.activation(attn_score)

        mask = top_rank(attn_score, graph_indicator, self.keep_ratio) #(334925)
        hidden = input_feature[mask] * attn_score[mask].view(-1, 1) #(167169,96) 注意*不同于矩阵乘法
        mask_graph_indicator = graph_indicator[mask] #(167169)
        mask_adjacency = filter_adjacency(adjacency, mask) #(167169,167169)
        return hidden, mask_graph_indicator, mask_adjacency

# 模型一： SAGPool Global Model
class ModelA(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构A

        Args:
        ----
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐藏层单元数
            num_classes: 分类类别数 (default: 2)
        """
        super(ModelA, self).__init__()
        self.input_dim = input_dim #89
        self.hidden_dim = hidden_dim #32
        self.num_classes = num_classes #2

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool = SelfAttentionPooling(hidden_dim * 3, 0.5)
        self.fc1 = nn.Linear(hidden_dim * 3 * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, adjacency, input_feature, graph_indicator):
        '''
        :param adjacency: (334925,334925)
        :param input_feature:
        :param graph_indicator:
        :return:
        '''
        gcn1 = F.relu(self.gcn1(adjacency, input_feature)) #(334925,32)
        gcn2 = F.relu(self.gcn2(adjacency, gcn1)) #(334925,32)
        gcn3 = F.relu(self.gcn3(adjacency, gcn2)) #(334925,32)

        gcn_feature = torch.cat((gcn1, gcn2, gcn3), dim=1) #(334925,96)
        pool, pool_graph_indicator, pool_adjacency = self.pool(adjacency, gcn_feature,
                                                               graph_indicator)

        readout = torch.cat((global_avg_pool(pool, pool_graph_indicator),
                             global_max_pool(pool, pool_graph_indicator)), dim=1) #(1178,192)

        fc1 = F.relu(self.fc1(readout)) #(1178,32)
        fc2 = F.relu(self.fc2(fc1)) #(1178,16)
        logits = self.fc3(fc2) #(1178,2)

        return logits

# 模型二：SAGPool Hierarchical Model
class ModelB(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes=2):
        """图分类模型结构

        Args:
        -----
            input_dim: int, 输入特征的维度
            hidden_dim: int, 隐藏层单元数
            num_classes: int, 分类类别数 (default: 2)
        """
        super(ModelB, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.gcn1 = GraphConvolution(input_dim, hidden_dim)
        self.pool1 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn2 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool2 = SelfAttentionPooling(hidden_dim, 0.5)
        self.gcn3 = GraphConvolution(hidden_dim, hidden_dim)
        self.pool3 = SelfAttentionPooling(hidden_dim, 0.5)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, adjacency, input_feature, graph_indicator):
        gcn1 = F.relu(self.gcn1(adjacency, input_feature))
        pool1, pool1_graph_indicator, pool1_adjacency = \
            self.pool1(adjacency, gcn1, graph_indicator)
        global_pool1 = torch.cat(
            [global_avg_pool(pool1, pool1_graph_indicator),
             global_max_pool(pool1, pool1_graph_indicator)],
            dim=1)

        gcn2 = F.relu(self.gcn2(pool1_adjacency, pool1))
        pool2, pool2_graph_indicator, pool2_adjacency = \
            self.pool2(pool1_adjacency, gcn2, pool1_graph_indicator)
        global_pool2 = torch.cat(
            [global_avg_pool(pool2, pool2_graph_indicator),
             global_max_pool(pool2, pool2_graph_indicator)],
            dim=1)

        gcn3 = F.relu(self.gcn3(pool2_adjacency, pool2))
        pool3, pool3_graph_indicator, pool3_adjacency = \
            self.pool3(pool2_adjacency, gcn3, pool2_graph_indicator)
        global_pool3 = torch.cat(
            [global_avg_pool(pool3, pool3_graph_indicator),
             global_max_pool(pool3, pool3_graph_indicator)],
            dim=1)

        readout = global_pool1 + global_pool2 + global_pool3

        logits = self.mlp(readout)
        return logits

if __name__ == '__main__':
    dataset=DDDataset()

    # 模型输入数据准备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    adjacency = dataset.sparse_adjacency #csr_matrix (334925,334925)
    normalize_adjacency = normalization(adjacency).to(DEVICE) # tensor (334925,334925)
    node_labels = tensor_from_numpy(dataset.node_labels, DEVICE)# tensor (334925)
    node_features = F.one_hot(node_labels, node_labels.max().item() + 1).float() # tensor（334925,89）
    graph_indicator = tensor_from_numpy(dataset.graph_indicator, DEVICE) #tensor (334925)
    graph_labels = tensor_from_numpy(dataset.graph_labels, DEVICE) #tensor(1178)
    train_index = tensor_from_numpy(dataset.train_index, DEVICE) #tensor (942)
    test_index = tensor_from_numpy(dataset.test_index, DEVICE) # tensor(236)
    train_label = tensor_from_numpy(dataset.train_label, DEVICE)#tensor (942)
    test_label = tensor_from_numpy(dataset.test_label, DEVICE) #tensor (236)

    # 超参数设置
    INPUT_DIM = node_features.size(1) #89
    NUM_CLASSES = 2
    EPOCHS = 2  # @param {type: "integer"}
    HIDDEN_DIM = 32  # @param {type: "integer"}
    LEARNING_RATE = 0.01  # @param
    WEIGHT_DECAY = 0.0001  # @param

    # 模型初始化
    model_g = ModelA(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    #model_h = ModelB(INPUT_DIM, HIDDEN_DIM, NUM_CLASSES).to(DEVICE)
    model=model_g
    print('device:',DEVICE)
    print('model:',model)

    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    model.train()
    for epoch in range(EPOCHS):
        logits = model(normalize_adjacency, node_features, graph_indicator)
        loss = criterion(logits[train_index], train_label)  # 只对训练的数据计算损失值
        optimizer.zero_grad()
        loss.backward()  # 反向传播计算参数的梯度
        optimizer.step()  # 使用优化方法进行梯度更新
        train_acc = torch.eq(
            logits[train_index].max(1)[1], train_label).float().mean()
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}".format(
            epoch, loss.item(), train_acc.item()))


    # 测试阶段
    model.eval()
    with torch.no_grad():
        logits=model(normalize_adjacency,node_features,graph_indicator)
        test_logits=logits[test_index]
        test_acc=torch.eq(test_logits.max(1)[1],test_label).float().mean()
    print('test_acc:',test_acc.item())