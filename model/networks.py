import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from model import utils 


class HyperGCN(nn.Module):
    def __init__(self, V, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        v:节点集合
        E：边集合
        x: 输入特征
        args: 属性参数
        """
        super(HyperGCN, self).__init__()
        # 这里初步猜测l为卷积层数，d是初始节点特征维数，也是卷积的大小，类别数是卷积核的个数。
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()
        # 隐藏节点的个数，和初始输入节点特征维数一致
        h = [d]
        # depth=2，两层卷积
        for i in range(l-1):
            # i可取值为0, power = 2 - 0 + 2 = 4
            power = l - i + 2
            # 如果数据集为'citesser', power = 6
            if args.dataset == 'citeseer': power = l - i + 4
            # 隐藏节点个数为2的power次方
            h.append(2**power)
        # 添加最后一个隐藏节点个数
        h.append(c)

        if args.fast:
            # 如果是fast模型，重新近似
            # 只使用初始特征X（没有权重）来构造超图拉普拉斯矩阵（有mediators）
            reapproximate = False
            structure = utils.Laplacian(V, E, X, args.mediators)        
        else:
            # 如果不是，用完整信息
            reapproximate = True
            structure = E
        #  网络层列表，
        self.layers = nn.ModuleList([utils.HyperGraphConvolution(h[i], h[i+1], reapproximate, cuda) for i in range(l)])
        self.do, self.l = args.dropout, args.depth
        self.structure, self.m = structure, args.mediators

    def forward(self, H):
        """
        an l-layer GCN
        """
        # dropout, layer number, mediators
        do, l, m = self.do, self.l, self.m
        # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
        for i, hidden in enumerate(self.layers):
            H = F.relu(hidden(self.structure, H, m))
            if i < l - 1:
                # 第一层需要dropout
                V = H
                H = F.dropout(H, do, training=self.training)
        
        return F.log_softmax(H, dim=1)
