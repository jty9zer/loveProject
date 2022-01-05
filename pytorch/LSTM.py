import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
import matplotlib.pyplot as plt

# 数据归一化处理工具
train_feature_scaler = MinMaxScaler(feature_range=(0, 1))
train_labels_scaler = MinMaxScaler(feature_range=(0, 1))
test_feature_scaler = MinMaxScaler(feature_range=(0, 1))
test_labels_scaler = MinMaxScaler(feature_range=(0, 1))


class LSTM(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.hidden_layer_size = n_hidden

        self.lstm = torch.nn.LSTM(n_input, n_hidden)

        self.linear = torch.nn.Linear(n_hidden, n_output)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


def load_data():
    # 读取数据
    data = pd.read_csv("../train_data.csv", sep=',', header=0)

    # 最后一列为label，其余为feature
    features_df = data.iloc[:, [-2]]
    label_df = data.iloc[:, [-1]]

    # 先切数据，再归一化处理, 避免使用未来数据
    # 分割训练数据和测试数据
    # trainX, testX, trainY, testY = train_test_split(features_df, label_df, train_size=0.8, random_state=1)
    trainY = label_df[:100]
    testY = label_df[100:]

    # 归一化数据
    # trainX = train_feature_scaler.fit_transform(trainX)
    # trainY = train_labels_scaler.fit_transform(trainY)
    trainY = train_labels_scaler.fit_transform(trainY)
    # testX = test_feature_scaler.fit_transform(testX)
    testY = test_labels_scaler.fit_transform(testY)

    # # 使用未来数据
    # # 归一化数据
    # features = feature_scaler.fit_transform(features_df)
    # labels = labels_scaler.fit_transform(label_df)
    #
    # # 截取训练数据
    # trainX, testX, trainY, testY = train_test_split(features, labels, train_size=0.8, random_state=1)

    # 将数据格式转换为tensor
    # trainX = torch.tensor(trainX).to(torch.float32)
    trainY = torch.tensor(trainY).to(torch.float32)
    # testX = torch.tensor(testX).to(torch.float32)
    testY = torch.tensor(testY).to(torch.float32)

    # trainX = torch.FloatTensor(trainX).view(-1)
    trainY = torch.FloatTensor(trainY).view(-1)
    # testX = torch.FloatTensor(testX).view(-1)
    testY = torch.FloatTensor(testY).view(-1)

    train_window = 7
    # trainX = create_inout_sequences(trainX, train_window)
    trainY = create_inout_sequences(trainY, train_window)
    # testX = create_inout_sequences(testX, train_window)
    testY = create_inout_sequences(testY, train_window)

    print("训练集大小：", len(trainY), "测试集大小：", len(testY))

    return trainY, testY


def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# 目标函数，返回模型的预测误差，可用于GA遗传算法的适应度函数。
def aim_function(threshold):
    # 读取数据
    trainY, testY = load_data()

    input_size = 1  # 输入层大小
    output_size = 1  # 输出层大小
    hidden_size = 7  # 隐藏层大小
    learning_rate = 0.001  # 学习速率
    loop_size = 50  # 训练次数

    # 定义一个 输入神经元为input_size、隐藏层神经元为hidden_size、输出神经元为output_size 的bp神经网络
    net = LSTM(input_size, hidden_size, output_size)
    # print("神经网络模型：", net)

    # 定义优化器
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Rprop(net.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)

    # 定义损失函数
    loss_func = torch.nn.L1Loss()  # MAE 平均绝对误差
    # loss_func = torch.nn.MSELoss()  # MSE 均方误差

    # 进行训练
    for step in range(loop_size):
        for seq, label in trainY:
            net.hidden_cell = (torch.zeros(1, 1, net.hidden_layer_size),
                               torch.zeros(1, 1, net.hidden_layer_size))
            prediction = net(seq)
            single_loss = loss_func(prediction, label)
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

        if step % (loop_size / 10) == 0:
            print("第 %d 次训练后, 训练集损失函数为：%g" % (step, single_loss.item()))

    train_loss = single_loss.item()

    result = []
    real_value = []
    # 进行预测
    for seq, label in testY:
        y_pred = net(seq)
        result.append(y_pred)
        real_value.append(label)
    result = torch.Tensor(result)
    real_value = torch.Tensor(real_value)
    single_loss = loss_func(result, real_value)

    print("训练结束, 当前神经网络的隐藏层层数为%d, 学习率设定为%f, 损失阈值设定为%f, 经过 %d 次训练后, 训练集损失为：%g, 测试集损失：%g" % (
        hidden_size, learning_rate, threshold, step, train_loss, single_loss.item()))

    # 对真实值和预测值进行反归一化处理，用于画图
    result = test_labels_scaler.inverse_transform(result.numpy().reshape(-1, 1))
    real_value = train_labels_scaler.inverse_transform(real_value.numpy().reshape(-1, 1))
    draw(real_value, result)

    return single_loss.item()


# 画图工具，画出真实值和预测值对应图
def draw(real_value, result):
    print("real value", real_value)
    print("predict result", result)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.plot(real_value, result, 'ro')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(0, 3000)
    plt.ylim(0, 3000)
    plt.show()


# 进行多次训练，画出训练次数和错误率的折线图
if __name__ == "__main__":
    train_times = 1  # 训练次数

    X = []
    Y = []
    for i in range(0, train_times):
        print("第%d次训练：" % i)
        X.append(i)
        Y.append(aim_function(0.07))
    # plt.plot(X, Y, 'r')
    # plt.show()
