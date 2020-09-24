# -*- coding:utf-8 -*-
"""
@Time: 2020/09/17 15:17
@Author: Shanshan Wang
@Version: Python 3.7
@Function:
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0),-1)

def load_protonet_conv(**kwargs):
    """
    :param kwargs:
    :return:
    """
    x_dim=kwargs['x_dim']
    hid_dim=kwargs['hid_dim']
    z_dim=kwargs['z_dim']

    def conv_block(in_channel,out_channel):
        return nn.Sequential(
            nn.Conv2d(in_channel,out_channel,3,padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
    encoder=nn.Sequential(
        conv_block(x_dim[0],hid_dim),
        conv_block(hid_dim,hid_dim),
        conv_block(hid_dim,hid_dim),
        conv_block(hid_dim,z_dim),
        Flatten()
    )
    return ProtoNet(encoder)

class ProtoNet(nn.Module):
    def __init__(self,encoder):
        super(ProtoNet, self).__init__()
        #self.encoder=encoder.cuda()
        self.encoder=encoder

    def set_forward_loss(self,sample):
        '''
        :param sample:
        :return:
        '''
        #sample_images=sample['images'].cuda()
        sample_images = sample['images'] # shape(n_way,n_support+n_query,3,28,28)
        n_way=sample['n_way']
        n_support=sample['n_support']
        n_query=sample['n_query']

        x_support=sample_images[:,:n_support] #(n_way,n_support,3,28,28)
        x_query=sample_images[:,n_support:]   #(n_way,n_query,3,28,28)

        target_inds=torch.arange(0,n_way).view(n_way,1,1).expand(n_way,n_query,1).long() #[n_way,n_support,1]
        #target_inds=target_inds.cuda()

        # 编码图片
        x=torch.cat([x_support.contiguous().view(n_way*n_support,*x_support.size()[2:]),
                     x_query.contiguous().view(n_way*n_support,*x_query.size()[2:])],0) #[n_way*(n_support+n_query),3,28,28]
        z=self.encoder.forward(x) #[n_way*(n_support+n_query),64]

        z_dim=z.size(-1) #64
        z_proto=z[:n_way*n_support].view(n_way,n_support,z_dim).mean(1) # [n_way,64]
        z_query=z[n_way*n_query:] #(n_way*n_query,64)

        # 计算距离
        dists=euclidean_dist(z_query,z_proto) #[n_way*n_query,n_way]

        # 计算概率
        log_p_y=F.log_softmax(-dists,dim=1).view(n_way,n_query,-1) #[n_way,n_query,n_way]

        loss_val=-log_p_y.gather(2,target_inds).squeeze().view(-1).mean() # scalar
        _,y_hat=log_p_y.max(2) # y_hat:[n_way,n_query] ,_:[n_way,n_query]
        acc_val=torch.eq(y_hat,target_inds.squeeze()).float().mean() # scalar

        return loss_val,{'loss':loss_val.item(),'acc':acc_val.item(),'y_hat':y_hat}

def euclidean_dist(x,y):
    '''
    :param x: query sample
    :param y: class prototype
    :return:
    '''
    n=x.size(0)
    m=y.size(0)
    d=x.size(1)
    assert d==y.size(1)

    x=x.unsqueeze(1).expand(n,m,d) # x.unsqueeze(1):(n,1,d)
    y=y.unsqueeze(0).expand(n,m,d) # y.unsqueeze(1):(1,m,d)
    return torch.pow(x-y,2).sum(2) # (n,m)
