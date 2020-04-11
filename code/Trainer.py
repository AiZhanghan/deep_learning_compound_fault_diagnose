import os
import time
import torch
import numpy as np


class Trainer:
    '''
    模型训练:
        train, 训练模型
        _evaluate_accuracy, 评价指标(准确率)
    '''

    def __init__(self, net, num_epochs, learning_rate, weight_decay,
                 batch_size, device):
        """
        Args:
            net: 待训练模型, torch中的网络
            num_epochs: int, 迭代次数
            learning_rate: float, 学习率
            weight_decay: float, TODO
            batch_size: int, TODO
            device: train on CPU or GPU
        """
        self.net = net
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.device = device
        print("train on", self.device)

    def train(self, log, train_iter, valid_iter, test_iter=None):
        '''训练模型
        Args:
            log: str, 日志
            train_iter: 训练集数据迭代器
            valid_iter: 验证集数据迭代器
            test_iter: 测试集数据迭代器
        Return:
            net: 训练好的模型
            records: 训练记录
        '''
        # 网络初始化
        self.net.apply(self._weight_init)
        self.net = self.net.to(self.device)
        self.net = self.net.float()
        # net输出层未经过sigmoid, loss选BCEWithLogitsLoss
        # 若已经使用sigmoid, loss选BCELoss
        loss = torch.nn.BCEWithLogitsLoss()
        # 保存训练过程acc变化
        records = {}
        records["train_acc"] = []
        records["valid_acc"] = []
        if test_iter:
            records["test_acc"] = []

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

            train_acc = self._evaluate_accuracy(train_iter)
            records["train_acc"].append(train_acc)
            valid_acc = self._evaluate_accuracy(valid_iter)
            records["valid_acc"].append(valid_acc)

            print('epoch %d, loss %.3f, train acc %.3f, valid acc %.3f, ' % 
                (epoch + 1, train_ls_sum / batch_count, train_acc, valid_acc),
                end='')

            if test_iter:
                test_acc = self._evaluate_accuracy(test_iter)
                records["test_acc"].append(test_acc)

                print('test acc %.3f, ' % test_acc, end='')

            print('time %.1f sec' % (time.time() - start))
        
        predict_dic = self._predict(valid_iter, test_iter)
        self._save(records, predict_dic, log)
        
        return self.net, records

    def _predict(self, valid_iter, test_iter):
        """预测验证集和测试集数据
        Args:
            valid_iter: 验证集数据迭代器
            test_iter: 测试集数据迭代器
        Return:
            predict_dic: dict{str: np.array}
        """
        self.net.eval()
        y_hat = np.array([])
        y = np.array([])

        with torch.no_grad():
            for x, y_batch in valid_iter:
                y_hat_batch = (self.net(x.to(self.device))).cpu().numpy()
                if y_hat.size == 0:
                    y_hat = y_hat_batch
                    y = y_batch
                else:
                    y_hat = np.concatenate((y_hat, y_hat_batch))
                    y = np.concatenate((y, y_batch))
            if test_iter:
                for x, y_batch in test_iter:
                    y_hat_batch = (self.net(x.to(self.device))).cpu().numpy()
                    y_hat = np.concatenate((y_hat, y_hat_batch))
                    y = np.concatenate((y, y_batch))
        self.net.train()
        predict_dic = {}
        predict_dic["y_hat"] = y_hat
        predict_dic["y"] = y
        return predict_dic
    
    def _save(self, records, predict_dic, log_):
        """保存训练过程acc, 训练好的模型, 验证集测试集预测值
        Args:
            records: dict[list]
            predict_dic: dict{str: np.array}
            log_: str, 辅助日志
        """
        if not os.path.exists("./file"):
            os.mkdir("./file")
        for key in records:
            records[key] = np.array(records[key])
        # 当前时间, 用于log
        now = time.asctime(time.localtime()).replace(":", " ")
        now = now.replace(" ", "_")
        log = "_%s_%d_%f_%f_%d_%s" % (log_, self.num_epochs, 
            self.learning_rate, self.weight_decay, self.batch_size, now)
        np.savez(os.path.join("./file", "acc_records" + log), **records)
        np.savez(os.path.join("./file", "predict_dic" + log), **records)
        # torch.save(self.net, os.path.join("./file", "model%s.pkl" % log))


    def _evaluate_accuracy(self, data_iter):
        '''评估指标，准确率'''
        # 标签一模一样才算是正确 
        self.net.eval()    
        acc_sum = 0.0
        n = 0
        with torch.no_grad():
            for x, y in data_iter:
                y_hat = (self.net(x.to(self.device)) >= 0.5).int()
                acc_sum += ((y_hat == y.to(self.device)).all(axis = 1).float()
                            .sum().cpu().item())
                n += y.shape[0]
        self.net.train()
        return acc_sum / n

    def _weight_init(self, m):
        '''根据网络层的不同定义不同的初始化方式'''
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv1d，使用相应的初始化方式   
        elif isinstance(m, torch.nn.Conv1d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out',
                nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, torch.nn.BatchNorm1d):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
