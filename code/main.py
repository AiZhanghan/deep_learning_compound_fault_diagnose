import torch
import numpy as np

import model
from Trainer import Trainer
from DataLoader import DataLoader


def partition_train_full(data_dic, label_dic, info, batch_size):
    """数据集划分(数据预处理, 乱序), 在完备数据集上训练
    Args:
        data_dic: dict{str: np.array}
        label_dic: dict{str: np.array}
        info: dict{str: int}
        batch_size: int
    Return:
        train_iter: 训练集数据迭代器
        valid_iter: 验证集数据迭代器
    """
    train_valid_data = np.zeros((info["single_fault_num"] + 
        info["compound_fault_num"], 3, info["signal_len"]))
    train_valid_label = np.zeros((info["single_fault_num"] + 
        info["compound_fault_num"], info["label_len"]))
    # 同时遍历data_dic, label_dic
    index_start = 0
    for key in data_dic:
        index = slice(index_start, index_start + data_dic[key].shape[0])
        train_valid_data[index] = data_dic[key]
        train_valid_label[index] = label_dic[key]
        index_start += data_dic[key].shape[0]
    train_valid_data = torch.from_numpy(train_valid_data)\
                       .type(torch.FloatTensor)
    train_valid_label = torch.from_numpy(train_valid_label)\
                        .type(torch.FloatTensor)
    # shuffle
    perm = torch.randperm(train_valid_data.shape[0])
    train_valid_data = train_valid_data[perm]
    train_valid_label = train_valid_label[perm]
    # radio = train / (train + valid)
    radio = 0.8
    train_index = slice(0, int(radio * train_valid_data.shape[0]))
    valid_index = slice(int(radio * train_valid_data.shape[0]),
                        train_valid_data.shape[0])
    x_train = train_valid_data[train_index]
    y_train = train_valid_label[train_index]
    x_valid = train_valid_data[valid_index]
    y_valid = train_valid_label[valid_index]
    # 数据预处理, 标准化
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_valid = (x_valid - mean) / std
    
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size)
    valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size)

    return train_iter, valid_iter


def main():
    # hyper-parameter, not including network structure
    num_epochs = 3
    lr = 0.001
    weight_decay = 0.01
    batch_size = 32
    # 网络结构
    stride_1 = 20
    padding_1 = 22
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 取数据, np.array
    data_loader = DataLoader(path="/home/aistudio/data/data29676")
    # 如果交叉验证, data_dic, label_dic需要放在内存里, 若内存吃紧, 再优化
    data_dic, label_dic, info = data_loader.get_frequency_data()
    # 划分训练集, 验证集, 测试集
    # 1. full class data for train
    train_iter, valid_iter = partition_train_full(data_dic, label_dic, info, 
                                                  batch_size)

    trainer = Trainer(model.WDCNN(stride_1, padding_1), num_epochs, lr, 
                     weight_decay, batch_size, device)
    trainer.train("2560_full", train_iter, valid_iter)
    
    # 2. single fault data for trian
    # train_iter, valid_iter, test_iter = partition_train_single(data_dic, 
    #     label_dic, info, batch_size)
    

    # trainer = Trainer(model.WDCNN(stride_1, padding_1), num_epochs, lr, 
    #                  weight_decay, batch_size, device)
    # trainer.train("2560_single", train_iter, valid_iter, test_iter)

    
if __name__ == "__main__":
    main()










# def partition_train_single(data_dic, label_dic, info, batch_size):
#     """数据集划分(数据预处理, 乱序), 仅在单一故障数据集上训练
#     Args:
#         data_dic: dict{str: np.array}
#         label_dic: dict{str: np.array}
#         info: dict{str: int}
#         batch_size: int
#     Return:
#         train_iter: 训练集数据迭代器
#         valid_iter: 验证集数据迭代器
#         test_iter: 测试集数据迭代器
#     """
#     # train, valid, 单一故障数据
#     single_fault_data = np.zeros((info["single_fault_num"], 3, 
#                                  info["signal_len"]))
#     single_fault_label = np.zeros((info["single_fault_num"], 
#                                   info["label_len"]))
#     # test, 复合故障数据
#     compound_fault_data = np.zeros((info["compound_fault_num"], 3, 
#                                     info["signal_len"]))
#     compound_fault_label = np.zeros((info["compound_fault_num"], 
#                                      info["label_len"]))
#     # 同时遍历data_dic, label_dic
#     single_index_start = 0
#     compound_index_start = 0
#     for key in data_dic:
#         if "Normal" in key:
#             index = slice(single_index_start, 
#                 single_index_start + data_dic[key].shape[0])
#             single_fault_data[index] = data_dic[key]
#             single_fault_label[index] = label_dic[key]
#             single_index_start += data_dic[key].shape[0]
#         else:
#             index = slice(compound_index_start,
#                 compound_index_start + data_dic[key].shape[0])
#             compound_fault_data[index] = data_dic[key]
#             compound_fault_label[index] = label_dic[key]
#             compound_index_start += data_dic[key].shape[0]
        
#     single_fault_data = torch.from_numpy(single_fault_data) \
#                         .type(torch.FloatTensor)
#     single_fault_label = torch.from_numpy(single_fault_label) \
#                          .type(torch.FloatTensor)
#     compound_fault_data = torch.from_numpy(compound_fault_data) \
#                           .type(torch.FloatTensor)
#     compound_fault_label = torch.from_numpy(compound_fault_label) \
#                          .type(torch.FloatTensor)

#     # shuffle on single fault data
#     perm = torch.randperm(single_fault_data.shape[0])
#     single_fault_data = single_fault_data[perm]
#     single_fault_label = single_fault_label[perm]
#     # radio = train / (train + valid)
#     radio = 0.8
#     train_index = slice(0, int(radio * single_fault_data.shape[0]))
#     valid_index = slice(int(radio * single_fault_data.shape[0]),
#                         single_fault_data.shape[0])
    
#     x_train = single_fault_data[train_index]
#     y_train = single_fault_label[train_index]
#     x_valid = single_fault_data[valid_index]
#     y_valid = single_fault_label[valid_index]
#     x_test = compound_fault_data
#     y_test = compound_fault_label
#     # 数据预处理, 标准化
#     mean = x_train.mean()
#     std = x_train.std()
#     x_train.sub_(mean).div_(std)
#     x_valid.sub_(mean).div_(std)
#     x_test.sub_(mean).div_(std)
    
#     train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
#     valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)
#     test_dataset = torch.utils.data.TensorDataset(x_test, y_test)
#     print(x_train.shape, x_valid.shape, x_test.shape)
#     train_iter = torch.utils.data.DataLoader(train_dataset, batch_size)
#     valid_iter = torch.utils.data.DataLoader(valid_dataset, batch_size)
#     test_iter = torch.utils.data.DataLoader(test_dataset, batch_size)

#     return train_iter, valid_iter, test_iter
    