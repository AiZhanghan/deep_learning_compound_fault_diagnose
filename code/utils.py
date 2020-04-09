import torch


def weight_init(m):
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


def get_stride_padding(l_in):
    '''辅助确定CNN模型结构, 根据信号长度求解stride, padding，
    Args:
        l_in: int, 样本信号长度
    '''
    l_out = 128
    kernel_size = 64

    res = []

    for stride in range(1, 100):
        for padding in range(100):
            if ((l_in + 2 * padding - kernel_size) / stride + 1) == l_out:
                res.append((stride, padding))
    print(res)
