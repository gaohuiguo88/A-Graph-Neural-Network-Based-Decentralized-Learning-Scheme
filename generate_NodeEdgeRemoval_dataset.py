import torch
import argparse
import copy
import random
parser = argparse.ArgumentParser()
parser.add_argument('--data_size', type=int, default=2500,
                    help='Size of the training data')
parser.add_argument('--num_node', type=int, default=100,
                    help='number of the node')
parser.add_argument('--r', type=float, default=0.2,
                    help='threshold')

args = parser.parse_args()

test_dataset = torch.load('Test_RGG_Size%d_Node%d_R%.1f.pt' % (args.data_size*0.2, args.num_node, args.r))

# 产生edege removal的测试数据
# for p in [x /1000 for x in range(0,150,25)]:
#     cp_test_dataset = copy.deepcopy(test_dataset)
#     for i in range(len(cp_test_dataset)):
#         edge_index = cp_test_dataset[i].edge_index
#         edge_index = edge_index.T[torch.randperm(int(edge_index.shape[1] * (1 - p)))].T
#         cp_test_dataset[i].edge_index = edge_index
#     torch.save(cp_test_dataset,'Test_RGG_Size%d_Node%d_R%.1f_EdgeRemoval%.3f.pt' % (args.data_size*0.2, args.num_node, args.r, p))

# 产生node removal的测试数据
def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)

for nodep in [x /100 for x in range(25,30,5)]:#5
    cp_test_dataset = copy.deepcopy(test_dataset)
    save_node_num = int((1-nodep)*args.num_node)
    map_save_node = torch.cat((torch.tensor([i for i in range(save_node_num)]),-1*torch.ones((args.num_node - save_node_num))),dim=0)

    for i in range(len(cp_test_dataset)):
        # 更改edge_index
        edge_index = cp_test_dataset[i].edge_index
        num_edge = len(edge_index[0])

        save_edge_index = torch.zeros((2,num_edge))
        for j in range(num_edge):
            save_edge_index[0][j] = map_save_node[int(edge_index[0][j])]
            save_edge_index[1][j] = map_save_node[int(edge_index[1][j])]

        j = 0
        while j < num_edge:
            if save_edge_index[0][j] == -1 or save_edge_index[1][j] == -1:
                save_edge_index = del_tensor_ele(save_edge_index.T,j)
                save_edge_index = save_edge_index.T
                num_edge -= 1
            else:
                j += 1

        cp_test_dataset[i].edge_index = save_edge_index

        # 更改data.x
        data_x = cp_test_dataset[i].x
        new_data_x = data_x[:save_node_num,:]

        cp_test_dataset[i].x = new_data_x

        # 更改data.y
        cp_test_dataset[i].y = sum(cp_test_dataset[i].x) * torch.ones(save_node_num) / save_node_num

    torch.save(cp_test_dataset,
               'Test_RGG_Size%d_Node%d_R%.1f_NodeRemoval%.2f.pt' % (args.data_size * 0.2, args.num_node, args.r, nodep))