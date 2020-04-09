from DataLoader import DataLoader


def main():
    # 取数据, np.array
    data_loader = DataLoader()
    data_dic, label_dic = data_loader.get_frequency_data()

    return data_dic, label_dic
