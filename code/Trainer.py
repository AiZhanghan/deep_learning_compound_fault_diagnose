import numpy as np
import torch
import time

import utils


class Trainer:
    '''
    模型训练：
    k_fold, k折交叉验证
        get_k_fold_data，返回第i折交叉验证时所需要的训练和测试数据，以及数据预处理
        weight_init， 模型初始化
        train， 训练模型
            evaluate_accuracy， 评价指标
            predict， 预测得到y_hat
    '''

    def __init__(self, net, k, data, label, test_data, test_label, num_epochs,
        learning_rate, weight_decay, batch_size, device):
        """
        Args:
            net: torch中的网络
            k: int, 交叉验证次数
            data: torch.tensor, 包括训练集和验证集
            label: torch.tensor, 包括训练集和验证集
            test_data: torch.tensor, 测试集, 测试模型在并发故障数据缺失下性能
            test_label: torch.tensor, 测试集, 测试模型在并发故障数据缺失下性能
            num_epochs: int, 每折迭代次数
            learning_rate: float, 学习率
            weight_decay: float, TODO
            batch_size: int, TODO
            device: train on CPU or GPU
        """
        self.net = net
        self.k = k
        self.data = data
        self.label = label
        self.test_data = test_data
        self.test_label = test_label
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        print("train on ", self.device)

    def k_fold(self):
        '''k折交叉验证'''
        train_acc_list = []
        test_acc_list = []
        # 用于制作混淆矩阵 TODO
        y_hat_test = np.array([])

        for i in range(self.k):
            train_iter, valid_iter, test_iter = self.get_k_fold_data(i)
            # 网络初始化
            self.net.apply(utils.weight_init)

            train_acc, test_acc, y_hat_test_fold = self.train(train_iter,
                valid_iter, test_iter)

            train_acc_list.append(train_acc[-1])
            test_acc_list.append(test_acc[-1])

            if y_hat_test.size == 0:
                y_hat_test = y_hat_test_fold
            else:
                y_hat_test = np.concatenate((y_hat_test, y_hat_test_fold))

            # if i == 0:
            #     plot(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', 
            #         range(1, num_epochs + 1), test_acc, ['train', 'test'])
        
        for i in range(self.k):
            print('fold %d, train acc %f, test acc %f' % 
                (i, train_acc_list[i], test_acc_list[i]))
        
        return sum(train_acc_list) / self.k, sum(test_acc_list) / self.k, y_hat_test


    def get_k_fold_data(self, i):
        '''返回第i折交叉验证时所需要的训练, 验证, 测试数据, 以及数据预处理
        Args:
            i: int, 第i折交叉验证
        Return:
            train_iter, 训练集数据迭代器
            valid_iter, 验证集数据迭代器
            test_iter, 测试集数据迭代器
        '''
        fold_size = self.data.shape[0] // self.k
        for j in range(self.k):
            idx = slice(j * fold_size, (j + 1) * fold_size)
            x_part = self.data[idx]
            y_part = self.label[idx]
            if j == i:
                x_valid = x_part
                y_valid = y_part
            elif x_train is None:
                x_train = x_part
                y_train = y_part
            else:
                x_train = torch.cat((x_train, x_part), dim = 0)
                y_train = torch.cat((y_train, y_part), dim = 0)
        # 数据预处理, 标准化
        mean = x_train.mean()
        std = x_train.std()
        x_train = (x_train - mean) / std
        x_valid = (x_valid - mean) / std
        
        train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
        valid_dataset = torch.utils.data.TensorDataset(x_valid, y_valid)

        train_iter = torch.utils.data.DataLoader(train_dataset, 
            self.batch_size, shuffle=True)
        valid_iter = torch.utils.data.DataLoader(valid_dataset,
            self.batch_size)
        
        if self.test_data:
            x_test = (self.test_data - mean) / std
            test_dataset = torch.utils.data.TensorDataset(x_test, 
                self.test_label)
            test_iter = torch.utils.data.DataLoader(test_dataset, 
                self.batch_size)

        return train_iter, valid_iter, (test_iter if self.test_data else None)

    def train(self, train_iter, valid_iter, test_iter):
        '''训练模型
        Args:
            train_iter, 训练集数据迭代器
            valid_iter, 验证集数据迭代器
            test_iter, 测试集数据迭代器
        Return:
            TODO
        '''
        self.net = self.net.to(self.device)
        # net输出层未经过sigmoid, loss选BCEWithLogitsLoss
        # 若已经使用sigmoid, loss选BCELoss
        loss = torch.nn.BCEWithLogitsLoss()

        train_acc_list = []
        valid_acc_list = []
        if test_iter:
            test_acc_list = []

        optimizer = torch.optim.Adam(params=self.net.parameters(), 
                                     lr=self.learning_rate, 
                                     weight_decay=self.weight_decay)
        
        for epoch in range(self.num_epochs):
            self.net.train()
            batch_count = 0
            # 一个epoch的train_loss_sum
            train_ls_sum = 0.0

            start = time.time()
            for x, y in train_iter:
                x = x.to(self.device)
                y = y.to(self.device)
                y_hat = self.net(x)

                l = loss(y_hat, y)
                optimizer.zero_grad()
                l.backward()
                optimizer.step()
                train_ls_sum += l.cpu().item()
                batch_count += 1

            train_acc = self.evaluate_accuracy(train_iter)
            train_acc_list.append(train_acc)
            valid_acc = self.evaluate_accuracy(valid_iter)
            valid_acc_list.append(valid_acc)
            
            print('epoch %d, loss %.3f, train acc %.3f, valid acc %.3f, ' % 
                (epoch + 1, train_ls_sum / batch_count, train_acc, valid_acc),
                end='')

            if test_iter:
                test_acc = self.evaluate_accuracy(test_iter)
                test_acc_list.append(test_acc)

                print('test acc %.3f, ' % test_acc, end='')

            print('time %.1f sec' % (time.time() - start))
        
        if test_iter is not None:  
            y_hat_test = predict(test_iter, net, device)
            return train_acc_list, test_acc_list, y_hat_test
        else:
            return train_acc_list


def evaluate_accuracy(data_iter, net, device=None):
    '''评估指标，准确率'''
    # 标签一模一样才算是正确 
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()    
    acc_sum = 0.0
    n = 0

    with torch.no_grad():
        for x, y in data_iter:
            assert isinstance(net, torch.nn.Module)

            acc_sum += (((net(x.to(device)) >= 0.5).int() == y.to(device))
                .all(axis = 1).float().sum().cpu().item())

            n += y.shape[0]
    net.train()
    return acc_sum / n


def predict(test_iter, net, device=None):
    '''预测'''
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    net.eval()
    y_hat = np.array([])

    with torch.no_grad():
        for x, _ in test_iter:
            y_hat_batch = (net(x.to(device)) >= 0.5).int().cpu().numpy()
            if y_hat.size == 0:
                y_hat = y_hat_batch
            else:
                y_hat = np.concatenate((y_hat, y_hat_batch))
    net.train()
    return y_hat

