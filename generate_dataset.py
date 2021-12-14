import torch
from torch_geometric.data import Data
import math
from tqdm import tqdm
import argparse


# 定义基本参数 包括节点个数 数据集大小
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=100,
                    help='number of the node')
parser.add_argument('--r', type=float, default=0.2,
                    help='threshold')

args = parser.parse_args()

# 对定义的节点个数，数据集大小，RGG的阈值 赋值
num_node = args.num_node
data_size = args.data_size
r = args.r

# 创建存储数据为列表的格式
data_list = []

# 根据定义的数据集大小产生随机几何图
for k in tqdm(range(data_size)):
    x = torch.rand(num_node)
    y = torch.rand(num_node)
    # edge_index用来存边的信息，[i,j]表示节点i到节点j有边相连
    edge_index = []
    for i in range(num_node):
        for j in range(i,num_node):
            if i != j:
                # 计算节点与节点之间的距离，如果节点之间的距离小于设定的阈值r，那么将两个节点相连
                d = math.sqrt(pow((x[i] - x[j]), 2) + pow((y[i] - y[j]), 2))
                if d <= r:
                    # 加入到edge_index里面
                    edge_index.append(torch.Tensor([i, j]))
                    edge_index.append(torch.Tensor([j, i]))
    edge_index = torch.tensor([t.long().numpy() for t in edge_index]).T

    # 这里数据节点的feature是1维的满足N(0,1)正态分布
    data_x = torch.randn(num_node, 1)
    # 这里Data数据形式存储在data_list里面，x是数据，y是对应的标签就是平均之后的值
    data_list.append(Data(x=data_x, y=(sum(data_x) * torch.ones(num_node)) / num_node, edge_index=edge_index))


torch.save(data_list,'./RGG_Size%d_Node%d_R%.1f.pt'%(data_size,num_node,r))




