import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset
import argparse
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, eigsh
import math
import os
from tqdm import tqdm
from torch_geometric.loader import DataLoader

# 定义基本参数 包括节点个数 K值 BatchSize
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=100,
                    help='number of the node')
parser.add_argument('--r', type=float, default=0.2,
                    help='threshold')
parser.add_argument('--K', type=int, default=20,
                    help='the size of the filter order')
parser.add_argument('--batch_size', type=int, default=10,
                    help='the size of a batch')
parser.add_argument('--epoch', type=int, default=250,
                    help='the number of total epochs')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='the number of total epochs')
parser.add_argument('--name', type=str, default='FIR',
                    help='the model type')
parser.add_argument('--checkpoint-path', default='checkpoints', type=str)


class GraphConvolution(Module):
    def __init__(self, K, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.K = K
        self.weight = Parameter(torch.FloatTensor(K, in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        out = []
        adj_ = adj
        for i in range(self.K):
            if i == 0:
                support = torch.mm(input, self.weight[i])
                out.append(support)
            else:
                tmp = torch.mm(adj_, input)
                support = torch.mm(tmp, self.weight[i])
                out.append(support)
                adj_ = torch.mm(adj_, adj)
        output = sum(out)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# 这里是定义GCN的网络结构
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, K, nclasses=1):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(K=K, in_features=nfeat, out_features=nhid)
        self.gc2 = GraphConvolution(K=K, in_features=nhid, out_features=nhid)
        self.lin = nn.Linear(nhid, nclasses)

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        x = F.relu(x)
        x = self.gc2(x, adj)
        x = F.relu(x)
        x = self.lin(x)
        return x


class FIR(nn.Module):
    def __init__(self, K):
        super(FIR, self).__init__()
        self.gc = GraphConvolution(K=K, in_features=1, out_features=1)

    def forward(self, x, adj):
        return self.gc(x, adj)


# 这个函数得到论文里面的S
def getNormalizedAdj(data):
    row = data.edge_index[0].cpu()
    col = data.edge_index[1].cpu()
    raw_data = torch.ones(data.edge_index.shape[1])
    adj = sp.coo_matrix((raw_data, (row, col)), shape=(data.x.shape[0], data.x.shape[0])).toarray()
    evals_large, evecs_large = eigsh(adj, 1, which='LM')
    adj = torch.Tensor(adj / evals_large)
    adj = adj.to(device)

    return adj


def init_model(K, name):
    if name == 'GNN':
        return GCN(1, 32, K).to(device)
    elif name == 'FIR':
        return FIR(K).to(device)
    else:
        raise ValueError('Please input GNN or FIR!')


# 训练和验证部分
if __name__ == '__main__':
    # 选择张量运算的设备 cpu或者是cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    # 训练GNN还是FIR
    net = init_model(args.K, args.name)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.99))

    dataset = torch.load('E:\\PycharmDocuments\\gnn_average_consensus\\RGG_Size%d_Node%d_R%.1f.pt' % (
    args.data_size, args.num_node, args.r))
    # 使用torch的random_split将数据集分割成训练集验证集(0.6,0.2)
    test_file_path = 'Test_RGG_Size%d_Node%d_R%.1f.pt' % (args.data_size * 0.2, args.num_node, args.r)

    if os.path.exists(test_file_path) == False:
        test_dataset = dataset[:-int(args.data_size * 0.2)]
        torch.save(test_dataset, test_file_path)
    else:
        test_dataset = torch.load(test_file_path)
    training_dataset, valid_dataset = torch.utils.data.random_split(dataset[:int(args.data_size * 0.8)],
                                                                    [int(args.data_size * 0.6),
                                                                     int(args.data_size * 0.2)])
    train_iterator = DataLoader(training_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
    valid_iterator = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)

    record_val_loss = 1e16
    train_loss_list = []
    val_loss_list = []
    for epoch in range(args.epoch):
        # 训练网络部分
        pbar = tqdm(train_iterator)
        train_loss = []
        for sents in pbar:
            net.train()
            optimizer.zero_grad()
            sents_adj = getNormalizedAdj(sents)
            output = net(sents.x, sents_adj).squeeze()
            loss = criterion(output, sents.y.squeeze())
            train_loss.append(loss.detach())
            loss.backward()
            optimizer.step()

            pbar.set_description(
                'Epoch: {};  Type: Train; Loss: {:.5f}'.format(
                    epoch + 1, loss, )
            )
        train_loss_per_epoch = sum(train_loss)/len(train_loss)

        # 验证网络性能部分
        pbar = tqdm(valid_iterator)
        val_loss = []
        for sents in pbar:
            sents.to(device)
            if net is not None:
                net.eval()
                sents_adj = getNormalizedAdj(sents)
                output = net(sents.x, sents_adj).squeeze()
                loss = criterion(output, sents.y.squeeze())
                val_loss.append(loss.detach())
                pbar.set_description(
                    'Epoch: {};  Type: VAL; Loss: {:.5f}'.format(
                        epoch + 1, loss, )
                )
        val_loss_per_epoch = sum(val_loss)/len(val_loss)

        if val_loss_per_epoch < record_val_loss:
            if not os.path.exists(args.checkpoint_path):
                os.makedirs(args.checkpoint_path)
            with open(args.checkpoint_path + '/checkpoint_{}_{}.pth'.format(str(epoch + 1).zfill(2), args.name),
                      'wb') as f:
                torch.save(net.state_dict(), f)
            record_val_loss = val_loss_per_epoch

        train_loss_list.append(train_loss_per_epoch)
        val_loss_list.append(val_loss_per_epoch)
    torch.save(train_loss_list, 'train_loss.pt')
    torch.save(val_loss_list, 'val_loss.pt')

# 在测试集上测试，这里如果是在edge removal,node removal的测试集上，就加载对应的test_dataset
# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     args = parser.parse_args()
#     criterion = nn.MSELoss()
#
#     # test_file_path = 'Test_RGG_Size%d_Node%d_R%.1f.pt' % (args.data_size * 0.2, args.num_node, args.r)
#     test_file_path = 'Test_RGG_Size%d_Node%d_R%.1f_NodeRemoval0.25.pt' % (args.data_size * 0.2, args.num_node, args.r)
#     test_dataset = torch.load(test_file_path)
#     test_iterator = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0, pin_memory=True)
#     pbar = tqdm(test_iterator)
#     test_loss = []
#     test_net = init_model(args.K,args.name)
#     test_net.load_state_dict(torch.load(args.checkpoint_path + '/checkpoint_{}_{}.pth'.format(str(41).zfill(2),args.name)))
#
#     for sents in pbar:
#         sents.to(device)
#         if test_net is not None:
#             test_net.eval()
#             sents_adj = getNormalizedAdj(sents)
#             output = test_net(sents.x, sents_adj).squeeze()
#             loss = criterion(output, sents.y.squeeze())
#             test_loss.append(loss.detach())
#             pbar.set_description(
#                 'Type: Test; Loss: {:.5f}'.format(
#                     loss, )
#             )
#     test_loss_per_epoch = sum(test_loss) / len(test_loss)
#     print(test_loss_per_epoch)
#     torch.save(test_loss,'test_loss.pt')
