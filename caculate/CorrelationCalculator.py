import pandas as pd


def get_correlation(method, path, label):
    data = pd.read_csv(path, sep=',', header=0)
    return data.corr(method=method)[label]


if __name__ == "__main__":
    # print("计算原始数据相关性：")
    # print(get_correlation("spearman", "../train_data_a.csv", "Y"), "\n")
    # print("计算第一次处理后的数据相关性：")
    # print(get_correlation("spearman", "../train_data_2.csv", "Y"), "\n")
    # print("计算第二次处理后的数据相关性：")
    # print(get_correlation("spearman", "../train_data_3.csv", "Y"), "\n")
    print("计算数据相关性：")
    print(get_correlation("spearman", "../train_data_0104.csv", "Y"), "\n")
