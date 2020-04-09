'''
1. frequent， fft， finished
2. 混淆矩阵，错误分析, finished
3. order track
4. 解释性
5. handed feature, finished
'''

import time
import pandas as pd
import numpy as np
import scipy
import torch
# from sklearn.metrics import confusion_matrix
# import sklearn
from torch import nn, optim
from matplotlib import pyplot as plt
from IPython import display


'''
加载数据：
load_data, 加载原始时域数据
    data_preprocessing，数据预处理，标准化
load_frequency_data, 加载频域数据
    load_data, 加载原始时域数据
    fft, FFT for signal matrix
    data_preprocessing，数据预处理，标准化
load_feature, 加载时域频域特征数据
    load_data, 加载原始时域数据
    get_tiem_feature, 提取时域特征
    get_frequency_feature， 提取频域特征
        fft_, FFT for one signal
    data_preprocessing，数据预处理，标准化
'''


def load_data(N=5120, root="F:\\workspace\\compound_fault\\data\\time\\",
    preprocessing=True, shuffle=True):
    '''加载数据'''
    folder_names = [
        "1_Normal_Normal", 
        "2_MissingTooth_Normal", 
        "3_RootCrack_Normal",
        "4_Surface_Normal", 
        "5_ChippedTooth_Normal", 
        "6_Normal_OuterRace",
        "7_Normal_InnerRace",
        "8_Normal_Ball",
        "9_MissingTooth_OuterRace",
        "10_MissingTooth_InnerRace",
        "11_MissingTooth_Ball",
        "12_RootCrack_OuterRace",
        "13_RootCrack_InnerRace",
        "14_RootCrack_Ball",
        "15_Surface_OuterRace",
        "16_Surface_InnerRace",
        "17_Surface_Ball",
        "18_ChippedTooth_OuterRace",
        "19_ChippedTooth_InnerRace",
        "20_ChippedTooth_Ball"
        ]
    
    columns = [
        'MissingTooth',
        'RootCrack', 
        'Surface', 
        'ChippedTooth',
        'OuterRace',
        'InnerRace',
        'Ball'
        ]
    # 字母序
    # columns.sort()
    to_multilabel = {key: value for value, key in enumerate(columns)}

    datas = np.array([])
    labels = np.array([])

    for i in range(len(folder_names)):
        # array, N, C, L
        sensor1 = pd.read_csv(root + folder_names[i] + '\\sensor1.csv').values
        sensor1 = sensor1.reshape(-1, 1, N)
        sensor2 = pd.read_csv(root + folder_names[i] + '\\sensor2.csv').values
        sensor2 = sensor2.reshape(-1, 1, N)
        sensor3 = pd.read_csv(root + folder_names[i] + '\\sensor3.csv').values
        sensor3 = sensor3.reshape(-1, 1, N)

        data = np.concatenate((sensor1, sensor2, sensor3), axis = 1)

        # label = np.zeros((data.shape[0], 8))
        label = np.zeros((data.shape[0], 7))
        # 构造Multilabel
        index = [to_multilabel[_class] 
            for _class in folder_names[i].split('_')[-2: ]
            if _class in to_multilabel.keys()]
        
        for i in range(len(index)):
            label[:, index[i]] = 1
        
        if len(datas) == 0:
            datas = data
            labels = label
        else:
            datas = np.concatenate((datas, data), axis = 0)
            labels = np.concatenate((labels, label), axis = 0) 
    # array to tensor
    datas = torch.from_numpy(datas).type(torch.FloatTensor)
    labels = torch.from_numpy(labels).type(torch.FloatTensor)

    if preprocessing:
        # 数据预处理
        datas = data_preprocessing(datas)
    # shuffle
    if shuffle:
        perm = torch.randperm(datas.shape[0])
        datas = datas[perm]
        labels = labels[perm]
    
    return datas, labels


def load_frequency_data(N=5120,
    root="F:\\workspace\\compound_fault\\data\\time\\",
    preprocessing=True, shuffle=True):
    '''加载频域数据'''
    # 加载时域数据
    data, label = load_data(N, root, False, shuffle)
    # FFT
    _, data = fft(N, data.numpy())
    # np.array -> tensor
    data = torch.from_numpy(data).type(torch.FloatTensor)
    # 数据预处理
    if preprocessing:
        data = data_preprocessing(data)

    return data, label


def fft(N, data, sample_frequency=5120):
    '''
    FFT for signal matrix
    data: numpy (batch_size, channel, length)
    '''
    # 采样周期
    T = 1 / sample_frequency

    x = np.linspace(0, N * T, N)
    # 快速傅里叶变换,取模
    yf = np.abs(np.fft.fft(data)) / ((len(x) / 2))
    # 由于对称性，只取一半区间
    yf = yf[:, :, : N // 2]

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    return xf, yf


def load_feature(N=5120, 
    root="F:\\workspace\\compound_fault\\data\\time\\",
    preprocessing=True, shuffle=True):
    '''加载时域频域特征数据'''
    # 加载时域数据
    data, label = load_data(N, root, False, shuffle)
    data = data.numpy()
    # # 获得频域数据
    # xf, yf = fft(N, data)
    # 提取时域特征
    feature_time = np.apply_along_axis(get_time_feature, -1, data)
    # 提取频域特征
    feature_frequency = np.apply_along_axis(get_frequency_feature, -1, data)
    # 合并
    # 传感器方向合并
    feature_time = feature_time.reshape(data.shape[0], -1)
    feature_frequency = feature_frequency.reshape(data.shape[0], -1)
    # 时域频域方向合并
    feature = np.concatenate([feature_time, feature_frequency], axis=1)
    # np.array -> tensor
    feature = torch.from_numpy(feature).type(torch.FloatTensor)
    # 数据预处理
    if preprocessing:
        feature = data_preprocessing(feature)

    return feature, label


def get_time_feature(x):
    '''提取时域特征'''
    N = len(x)

    mean = np.mean(x)
    std = np.std(x)
    root = (np.sum(np.sqrt(np.abs(x))) / N) ** 2
    rms = np.sqrt(np.sum(x ** 2) / N)
    peak = np.max(np.abs(x))

    skewness = (np.sum((x - mean) ** 3)) / ((N - 1) * std ** 3)
    kurtosis = (np.sum((x - mean) ** 4)) / ((N - 1) * std ** 4)
    crest = peak / rms
    clearance = peak / root
    shape = rms / (np.sum(np.abs(x)) / N)

    impluse = peak / (np.sum(np.abs(x)) / N)

    feature =  np.array([[mean, std, root, rms, peak, skewness, kurtosis, # 7
        crest, clearance, shape, impluse]]) # 4

    return feature


def get_frequency_feature(x):
    '''提取频域特征'''
    # 采样点数
    N = len(x)
    xf, yf = fft_(N, x)

    K = len(yf)
    s = yf # spectrum
    f = xf # frequency value

    p1 = np.sum(s) / K
    p2 = np.sum((s - p1) ** 2) / (K - 1)
    p3 = np.sum((s - p1) ** 3) / (K * (np.sqrt(p2) ** 3))
    p4 = np.sum((s - p1) ** 4) / (K * p2 ** 2)
    p5 = np.sum(f * s) / np.sum(s)
    p6 = np.sqrt(np.sum((f - p5) ** 2 * s) / K)
    p7 = np.sqrt(np.sum(f ** 2 * s) / np.sum(s))
    p8 = np.sqrt(np.sum(f ** 4 * s) / np.sum(f ** 2 * s))
    p9 = np.sum(f ** 2 * s) / np.sqrt(np.sum(s) * np.sum(f ** 4 * s))
    p10 = p6 / p5
    p11 = np.sum((f - p5) ** 3 * s) / (K * p6 ** 3)
    p12 = np.sum((f - p5) ** 4 * s) / (K * p6 ** 4)
    p13 = np.sum(np.sqrt(np.abs(f - p5)) * s) / (K * np.sqrt(p6))
    p14 = np.sqrt(np.sum((f - p5) ** 2 * s) / np.sum(s))

    feature =  np.array([[p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12,
        p13, p14]])

    return feature


def fft_(N, data, sample_frequency=5120):
    '''
    FFT for one signal
    data: 1d array
    '''
    # 采样周期
    T = 1 / sample_frequency

    x = np.linspace(0, N * T, N)
    # 快速傅里叶变换,取模
    yf = np.abs(np.fft.fft(data)) / ((len(x) / 2))
    # 由于对称性，只取一半区间
    yf = yf[: N // 2]

    xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)

    return xf, yf


def data_preprocessing(datas):
    '''数据预处理, 标准化'''
    mean = datas.mean()
    std = datas.std()

    datas = (datas - mean) / std
    return datas


'''
模型训练：
k_fold, k折交叉验证
    get_k_fold_data，返回第i折交叉验证时所需要的训练和测试数据，以及数据预处理
    weight_init， 模型初始化
    train， 训练模型
        evaluate_accuracy， 评价指标
        predict， 预测得到y_hat
    plot， 训练过程可视化
        set_figsize, 设置图的尺寸
            use_svg_display, Use svg format to display plot in jupyter
get_stride_padding, 辅助确定CNN模型结构
'''


def k_fold(net, k, data, label, 
    num_epochs, learning_rate, weight_decay, batch_size, device):
    '''k折交叉验证'''
    train_acc_list = []
    test_acc_list = []
    # 用于制作混淆矩阵
    y_hat_test = np.array([])

    for i in range(k):
        train_iter, test_iter = get_k_fold_data(k, i, data, label,
            batch_size)
        # net = WDCNN()
        net.apply(weight_init)

        train_acc, test_acc, y_hat_test_fold = train(net, 
            train_iter, test_iter, num_epochs, 
            learning_rate, weight_decay, device)

        train_acc_list.append(train_acc[-1])
        test_acc_list.append(test_acc[-1])

        if y_hat_test.size == 0:
            y_hat_test = y_hat_test_fold
        else:
            y_hat_test = np.concatenate((y_hat_test, y_hat_test_fold))

        if i == 0:
            plot(range(1, num_epochs + 1), train_acc, 'epochs', 'acc', 
                range(1, num_epochs + 1), test_acc, ['train', 'test'])
    
    for i in range(k):
        print('fold %d, train acc %f, test acc %f' % 
            (i, train_acc_list[i], test_acc_list[i]))
    
    return sum(train_acc_list) / k, sum(test_acc_list) / k, y_hat_test


def get_k_fold_data(k, i, data, label, batch_size):
    '''返回第i折交叉验证时所需要的训练和测试数据，以及数据预处理'''
    assert k > 1
    fold_size = data.shape[0] // k
    x_train = None
    y_train = None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        x_part = data[idx, :]
        y_part = label[idx]
        if j == i:
            x_test = x_part
            y_test = y_part
        elif x_train is None:
            x_train = x_part
            y_train = y_part
        else:
            x_train = torch.cat((x_train, x_part), dim = 0)
            y_train = torch.cat((y_train, y_part), dim = 0)

    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size)

    return train_iter, test_iter


def weight_init(m):
    '''根据网络层的不同定义不同的初始化方式'''
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
    # 也可以判断是否为conv1d，使用相应的初始化方式   
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out',
            nonlinearity='relu')
    # 是否为批归一化层
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def train(net, train_iter, test_iter, num_epochs,
            learning_rate, weight_decay, device):
    '''训练模型'''
    net = net.to(device)
    net = net.float()
    # net.train()
    print("training on ", device)
    
    # loss = torch.nn.BCELoss()
    loss = torch.nn.BCEWithLogitsLoss()

    train_acc_list = []
    test_acc_list = []

    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, 
        weight_decay=weight_decay)
    batch_count = 0
    for epoch in range(num_epochs):
        net.train()
        train_ls_sum = 0.0

        start = time.time()
        for x, y in train_iter:
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)

            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_ls_sum += l.cpu().item()

            batch_count += 1
        # net.eval()    
        train_acc = evaluate_accuracy(train_iter, net, device)
        train_acc_list.append(train_acc)
        
        print('epoch %d, loss %.4f, train acc %.3f, ' % 
            (epoch + 1, train_ls_sum / batch_count, train_acc), end='')

        if test_iter is not None:
            test_acc = evaluate_accuracy(test_iter, net, device)
            test_acc_list.append(test_acc)

            print('test acc %.3f, ' % test_acc, end='')

        print('time %.1f sec' % (time.time() - start))
        # net.train()
        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
        #     % (epoch + 1, train_ls_sum / batch_count, train_acc, 
        #     test_acc, time.time() - start))
    
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


def plot(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
        legend=None, figsize=(3.5, 2.5)):
    '''训练过程可视化'''
    set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x_vals, y_vals)
    if x2_vals and y2_vals:
        plt.plot(x2_vals, y2_vals, linestyle=':')
        plt.legend(legend)
    plt.show()


def set_figsize(figsize=(3.5, 2.5)):
    '''设置图的尺寸'''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')


def get_stride_padding(l_in):
    '''根据信号长度求解stride, padding'''
    l_out = 128
    kernel_size = 64

    res = []

    for stride in range(1, 100):
        for padding in range(100):
            if ((l_in + 2 * padding - kernel_size) / stride + 1) == l_out:
                res.append((stride, padding))
    print(res)


'''
error analyze：
confusion_matrix 混淆矩阵
    plot_confusion_matrix， 可视化混淆矩阵
        plot_confusion_matrix_， 绘制
    get_confusion_df， get_confusion_df, pred_label, 涵盖所有结果
        multilabel2str
'''


def confusion_matrix(y_hat, label, figsize=(12, 8)):
    '''
    混淆矩阵
    confusion_matrix： 用于绘制可视化图像，
        未在true_label中的pred_label用other表示
    confusion_df: for error analyse,
        未在true_label中的pred_label具体表示出来
    '''
    # 由于K折交叉验证y_hat < label
    y = label.numpy()[: len(y_hat)]
    y = y.astype(np.int)

    cm = get_confusion_matrix(y, y_hat)
    plot_confusion_matrix(cm, figsize)
    
    return cm


def get_confusion_matrix(y, y_hat):
    '''get_confusion_df, pred_label, 涵盖所有结果'''
    columns = [
        "MT", # MissingTooth
        "RC", # RootCrack
        "S", # Surface
        "CT", # ChippedTooth
        "OR", # OuterRace
        "IR", # InnerRace
        "B" # Ball
        ]
    sort_weight = {key: value for value, key in enumerate(columns, 1)}
    sort_weight['N'] = 0 # Normal
    # columns.sort()

    y_df = pd.DataFrame(y, columns=columns)
    y_df = y_df.apply(multilabel2str, axis=1)

    y_hat_df = pd.DataFrame(y_hat, columns=columns)
    y_hat_df = y_hat_df.apply(multilabel2str, axis=1)

    index = sorted(y_df.unique(), 
        key=lambda x: (len(x.split('_')), sort_weight[x.split('_')[0]]))

    columns = sorted(y_hat_df.unique(), 
        key=lambda x: (x not in index, len(x.split('_')), 
            sort_weight[x.split('_')[0]]))

    confusion_df = pd.DataFrame(np.zeros((len(index), len(columns)), 
        dtype=np.int), index=index, columns=columns)

    for i in range(len(y_df)):
        confusion_df.loc[y_df.iloc[i], y_hat_df.iloc[i]] += 1
    
    return confusion_df


def multilabel2str(x):
    '''multilabel -> str'''
    index = x.index
    res = '_'.join(list(index[x == 1]))
    if not res:
        res = 'N'
    return res
    

def plot_confusion_matrix(cm, figsize):
    '''可视化混淆矩阵'''
    set_figsize(figsize)
    
    pred_labels = list(cm.columns)
    true_labels = list(cm.index)

    cm = cm.values
    
    x_tick_marks = np.array(range(len(pred_labels))) + 0.5
    y_tick_marks = np.array(range(len(true_labels))) + 0.5
    # tick_marks = np.array(range(len(labels))) + 0.5
    np.set_printoptions(precision=2)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    x_ind_array = np.arange(len(pred_labels))
    y_ind_array = np.arange(len(true_labels))

    x, y = np.meshgrid(x_ind_array, y_ind_array)

    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm_normalized[y_val][x_val]
        if c and c >= 0.01:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, 
                va='center', ha='center')
    
    # offset the tick
    plt.gca().set_xticks(x_tick_marks, minor=True)
    plt.gca().set_yticks(y_tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    _plot_confusion_matrix(cm_normalized, pred_labels, true_labels, 
        title='Normalized confusion matrix')
    
    plt.show()


def _plot_confusion_matrix(cm, pred_labels, true_labels,
    title='Confusion Matrix', cmap=plt.cm.get_cmap('Greys')):
    '''for func plot_confusion_matrix, 外部不该调用该函数'''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(pred_labels)))
    ylocations = np.array(range(len(true_labels)))
    plt.xticks(xlocations, pred_labels, rotation=90)
    plt.yticks(ylocations, true_labels)
    plt.ylim(-0.5, 19.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    '''for test'''

