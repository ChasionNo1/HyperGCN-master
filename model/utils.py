import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class HyperGraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    # a，b权重矩阵大小
    def __init__(self, a, b, reapproximate=True, cuda=True):
        super(HyperGraphConvolution, self).__init__()
        self.a, self.b = a, b
        self.reapproximate, self.cuda = reapproximate, cuda

        self.W = Parameter(torch.FloatTensor(a, b))
        self.bias = Parameter(torch.FloatTensor(b))
        self.reset_parameters()

    # 残差连接参数
    def reset_parameters(self):
        # 权重和偏置的初始化
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    # 前向传播
    def forward(self, structure, H, m=True):
        """
        structure:E
        H；图拉普拉斯矩阵
        m:使用中介
        特征和权重矩阵乘法,再和超图拉普拉斯做稀疏矩阵乘法,
        """
        W, b = self.W, self.bias
        # 矩阵乘法
        HW = torch.mm(H, W)

        if self.reapproximate:
            n, X = H.shape[0], HW.cpu().detach().numpy()
            A = Laplacian(n, structure, X, m)
        else: A = structure

        if self.cuda: A = A.cuda()
        A = Variable(A)

        AHW = SparseMM.apply(A, HW)     
        return AHW + b

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'


class SparseMM(torch.autograd.Function):
    """
    Sparse x dense matrix multiplication with autograd support.
    Implementation by Soumith Chintala:
    https://discuss.pytorch.org/t/
    does-pytorch-support-autograd-on-sparse-matrix/6156/7
    """
    @staticmethod
    def forward(ctx, M1, M2):
        ctx.save_for_backward(M1, M2)
        return torch.mm(M1, M2)

    @staticmethod
    def backward(ctx, g):
        M1, M2 = ctx.saved_tensors
        g1 = g2 = None

        if ctx.needs_input_grad[0]:
            g1 = torch.mm(g, M2.t())

        if ctx.needs_input_grad[1]:
            g2 = torch.mm(M1.t(), g)

        return g1, g2


def Laplacian(V, E, X, m):
    """
    approximates the E defined by the E Laplacian with/without mediators
    近似由E拉普拉斯算子定义的E（带有/不带有介体）

    arguments:
    V: number of vertices    顶点数量
    超边字典，key是超边，value是节点集合
    E: dictionary of hyperedges (key: hyperedge, value: list/set of hypernodes)
    X: features on the vertices
    m: True gives Laplacian with mediators, while False gives without
    图近似 邻接矩阵
    A: adjacency matrix of the graph approximation
    returns:
    以“ graph”为键更新数据，其值近似为超图
    updated data with 'graph' as a key and its value the approximated hypergraph 
    """

    """
    超图拉普拉斯矩阵的构造：
    1、对每个超边，取两个节点特征欧式矩阵最远的两项，代表这个超边
    2、给这个超边（两个节点）设置权值
    3、对称归一化超图拉普拉斯
    """
    # 这里是简化后的超边和对应权重的容器
    edges, weights = [], {}
    # 随意打破连接
    rv = np.random.rand(X.shape[1])
    # 对E中的超边进行遍历
    for k in E.keys():
        # 将超边组合成一个列表
        hyperedge = list(E[k])
        # 向量点积和矩阵乘法，超边的特征矩阵，与随机打断做矩阵乘法，然后选择两个欧式距离最远的特征
        p = np.dot(X[hyperedge], rv)   #projection onto a random vector rv 投影到随机向量rv上
        # 两个节点欧式距离最远的
        s, i = np.argmax(p), np.argmin(p)
        # 两个超边
        Se, Ie = hyperedge[s], hyperedge[i]

        # two stars with mediators，权重参数1/（2|e|-3）
        c = 2*len(hyperedge) - 3    # normalisation constant
        if m:
            # 带中介
            # connect the supremum (Se) with the infimum (Ie) 连接最大值和最小值
            edges.extend([[Se, Ie], [Ie, Se]])
            # 有向图啊？
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/c)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/c)
            
            # connect the supremum (Se) and the infimum (Ie) with each mediator
            for mediator in hyperedge:
                if mediator != Se and mediator != Ie:
                    # 添加，四个有向边
                    edges.extend([[Se,mediator], [Ie,mediator], [mediator,Se], [mediator,Ie]])
                    # 更新权重
                    weights = update(Se, Ie, mediator, weights, c)
        # 不使用中介
        else:
            edges.extend([[Se,Ie], [Ie,Se]])
            e = len(hyperedge)
            
            if (Se,Ie) not in weights:
                weights[(Se,Ie)] = 0
            weights[(Se,Ie)] += float(1/e)

            if (Ie,Se) not in weights:
                weights[(Ie,Se)] = 0
            weights[(Ie,Se)] += float(1/e)

    return adjacency(edges, weights, V)


def update(Se, Ie, mediator, weights, c):
    """
    updates the weight on {Se,mediator} and {Ie,mediator}
    """    
    
    if (Se,mediator) not in weights:
        weights[(Se,mediator)] = 0
    weights[(Se,mediator)] += float(1/c)

    if (Ie,mediator) not in weights:
        weights[(Ie,mediator)] = 0
    weights[(Ie,mediator)] += float(1/c)

    if (mediator,Se) not in weights:
        weights[(mediator,Se)] = 0
    weights[(mediator,Se)] += float(1/c)

    if (mediator,Ie) not in weights:
        weights[(mediator,Ie)] = 0
    weights[(mediator,Ie)] += float(1/c)

    return weights


def adjacency(edges, weights, n):
    """
    computes an sparse adjacency matrix，计算稀疏邻接矩阵

    arguments:
    edges: list of pairs
    weights: dictionary of edge weights (key: tuple representing edge, value: weight on the edge)
    n: number of nodes

    returns: a scipy.sparse adjacency matrix with unit weight self loops for edges with the given weights
    """
    # {0:(item1),1:(item2),....}
    dictionary = {tuple(item): index for index, item in enumerate(edges)}
    # 得到边的索引列表
    edges = [list(itm) for itm in dictionary.keys()]
    # 权重列表
    organised = []
    # 给边加权重，
    for e in edges:
        # 一个边的两个节点
        i,j = e[0],e[1]
        w = weights[(i,j)]
        organised.append(w)
    # 将list转换为np数组
    edges, weights = np.array(edges), np.array(organised)
    # 稀疏矩阵的格式
    adj = sp.coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=np.float32)
    # 加上对角矩阵,稀疏形式 (0,0)  1
    adj = adj + sp.eye(n)

    A = symnormalise(sp.csr_matrix(adj, dtype=np.float32))
    A = ssm2tst(A)
    return A


def symnormalise(M):
    """
    对称正则化稀疏矩阵
    symmetrically normalise sparse matrix

    arguments:
    M: scipy sparse matrix

    returns:
    D^{-1/2} M D^{-1/2} 
    where D is the diagonal node-degree matrix
    """
    # 度矩阵
    d = np.array(M.sum(1))
    # 度矩阵的-1/2次方,再平铺
    dhi = np.power(d, -1/2).flatten()
    dhi[np.isinf(dhi)] = 0.
    DHI = sp.diags(dhi)    # D half inverse i.e. D^{-1/2}
    
    return (DHI.dot(M)).dot(DHI) 


def ssm2tst(M):
    """
    converts a scipy sparse matrix (ssm) to a torch sparse tensor (tst)
    将scipy稀疏矩阵（ssm）转换为torch稀疏张量（tst）

    arguments:
    M: scipy sparse matrix

    returns:
    a torch sparse tensor of M
    """
    
    M = M.tocoo().astype(np.float32)
    
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long()
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
