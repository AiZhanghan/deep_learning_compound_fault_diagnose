import os
import time
import numpy as np
import pandas as pd


class DataLoader:
    '''数据加载器
    get_time_data: 获取时域数据
    get_frequency_data: 获取频域数据
    load_data: csv to npz, 加载原始时域数据
    '''

    def __init__(self, N=5120,
        path="D:/Workspace/Data/20191113_compound_fault"):
        """
        Args:
            N: int, 样本长度
            path: str, 目标文件夹路径
        """
        self.N = N
        self.path = path

    def get_time_data(self):
        """获取时域数据
        Return:
            data: dict{str: np.array}
            label: dict{str: np.array}
        """
        data = np.load(os.path.join(self.path, "data_%s.npz" % self.N))
        label = np.load(os.path.join(self.path, "label_%s.npz" % self.N))
        return dict(data), dict(label)

    def get_frequency_data(self):
        '''获取频域数据
        Return:
            data: dict{str: np.array}
            label: dict{str: np.array}
        '''
        # 获取时域数据
        data, label = self.get_time_data()
        # FFT
        for key in data:
            _, data[key] = self._fft(data[key])

        return data, label

    def load_data(self, 
        source_path=r"D:/Workspace/Data/20191113_compound_fault/time"):
        '''csv to npz, 加载时域数据
        Args:
            source_path: str, 源数据文件夹路径
        '''
        # 得到source_path目录下所有文件夹并排序
        folders = os.listdir(source_path)
        folders.sort(key=lambda x: int(x.split("_")[0]))
        filenames = ["sensor1.csv", "sensor2.csv", "sensor3.csv"]
        
        columns = [
            'MissingTooth',
            'RootCrack', 
            'Surface', 
            'ChippedTooth',
            'OuterRace',
            'InnerRace',
            'Ball'
            ]
        to_multilabel = {key: value for value, key in enumerate(columns)}
        # datas, 包括所有样本
        # data, 单类样本, 用于组建datas
        datas = {}
        labels = {}

        for folder in folders:
            print("loading %s" % folder, end="\t")
            start_time = time.time()
            # 合并三个传感器为三个通道(batch_size, C, N)
            path = os.path.join(source_path, folder, filenames[0])
            data = pd.read_csv(path).values.reshape(-1, 1, self.N)
            for filename in filenames[1: ]:
                path = os.path.join(source_path, folder, filename)
                sensor = pd.read_csv(path).values.reshape(-1, 1, self.N)
                data = np.concatenate((data, sensor), axis = 1)
            
            label = np.zeros((data.shape[0], 7))
            # 构造Multilabel
            index = [to_multilabel[_class] 
                     for _class in folder.split('_')[-2: ]
                     if _class in to_multilabel]
            for i in range(len(index)):
                label[:, index[i]] = 1
            
            datas[folder] = data.copy()
            labels[folder] = label.copy()

            print("%.2fs" % (time.time() - start_time))

        np.savez(os.path.join(path, "data_%s" % self.N), **datas)
        np.savez(os.path.join(path, "label_%s" % self.N), **labels)


    def _fft(self, data, sample_frequency=5120):
        '''FFT for signal matrix
        Args:
            data: np.array, (batch_size, channel, length)
            sample_frequency: 采样频率
        Return:
            xf: np.array, 对应频率(Hz)
            yf: np.array, 幅值(未预处理)
        '''
        # 采样周期
        T = 1 / sample_frequency
        x = np.linspace(0, self.N * T, self.N)
        # 快速傅里叶变换,取模
        yf = np.abs(np.fft.fft(data)) / ((len(x) / 2))
        # 由于对称性，只取一半区间
        yf = yf[:, :, : self.N // 2]
        xf = np.linspace(0.0, 1.0 / (2.0 * T), self.N // 2)

        return xf, yf


def main():
    """test"""
    data_loader = DataLoader()
    data_loader.load_data()


if __name__ == "__main__":
    main()
