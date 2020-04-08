'''
model:
1. WDCNN
2. MLP
3. WDCNN_logit, TODO
'''


from torch import nn


class WDCNN(nn.Module):
    '''WDCNN'''
    '''
    根据输入数据长度调整模型结构
    {   
        # length: (stride, padding)
        5120：(40, 12),
        2560: (20, 22)
    }
    '''
    def __init__(self, stride_1, padding_1):
        super(WDCNN, self).__init__()
        # input_size = (batch_size, 3, 5120), (N, Cin, Lin)
        # need padding
        self.conv = nn.Sequential(
            # layer1
            # in_channels, out_channels, kernel_size, stride
            nn.Conv1d(3, 16, 64, stride_1, padding=padding_1), 
            nn.BatchNorm1d(16), # BN在激活函数之前
            nn.ReLU(),
            nn.MaxPool1d(2, 2), # kernel_size, stride
            # layer2
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            # layer3
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            # layer4
            nn.Conv1d(64, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            # layer5
            nn.Conv1d(64, 64, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(3 * 64, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5), # Dropout在激活函数之后
            nn.Linear(128, 7),
            # nn.Sigmoid()
        )

    def forward(self, data):
        
        feature = self.conv(data)
        output = self.fc(feature.view(data.shape[0], -1))
        
        return output


class MLP(nn.Module):
    '''MLP'''
    def __init__(self, num_inputs):
        super(MLP, self).__init__()
        # input_size = (batch_size, (11 + 14) * 3)
        self.net = nn.Sequential(
            nn.Linear(num_inputs, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 7),
            nn.Sigmoid()
        )

    def forward(self, data):

        output = self.net(data)

        return output
    