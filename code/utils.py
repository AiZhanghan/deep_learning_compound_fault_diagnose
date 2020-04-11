import torch


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
