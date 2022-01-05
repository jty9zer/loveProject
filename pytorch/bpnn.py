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


# feature_scaler = MinMaxScaler(feature_range=(0, 1))
# labels_scaler = MinMaxScaler(feature_range=(0, 1))

# 定义神经网络
class Net(torch.nn.Module):
    # 定义网络结构
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 此为隐藏层到输出层

    # 定义前向传播函数
    def forward(self, input):
        out = self.predict(torch.nn.functional.relu(self.hidden(input)))  #
        # out = self.predict(torch.nn.functional.sigmoid(self.hidden1(input)))
        # out = self.predict(torch.nn.functional.tanh(self.hidden1(input)))
        return out

# 导入数据
def load_data():
    # 读取数据
    data = pd.read_csv("../train_data_0104.csv", sep=',', header=0)

    # 最后一列为label，其余为feature
    features_df = data.iloc[:, :-1]
    label_df = data.iloc[:, [-1]]

    # 先切数据，再归一化处理, 避免使用未来数据
    # 分割训练数据和测试数据
    trainX, testX, trainY, testY = train_test_split(features_df, label_df, train_size=0.8, random_state=1)

    # 归一化数据
    trainX = train_feature_scaler.fit_transform(trainX)
    trainY = train_labels_scaler.fit_transform(trainY)
    testX = test_feature_scaler.fit_transform(testX)
    testY = test_labels_scaler.fit_transform(testY)

    # # 使用未来数据
    # # 归一化数据
    # features = feature_scaler.fit_transform(features_df)
    # labels = labels_scaler.fit_transform(label_df)
    #
    # # 截取训练数据
    # trainX, testX, trainY, testY = train_test_split(features, labels, train_size=0.8, random_state=1)

    # 将数据格式转换为tensor
    trainX = torch.tensor(trainX).to(torch.float32)
    trainY = torch.tensor(trainY).to(torch.float32)
    testX = torch.tensor(testX).to(torch.float32)
    testY = torch.tensor(testY).to(torch.float32)

    # print("训练集大小：", len(trainY), "测试集大小：", len(testY))

    return trainX, trainY, testX, testY


# 目标函数，返回模型的预测误差，可用于GA遗传算法的适应度函数。
def aim_function(size, eta, threshold, max_loop=2000):
    # 读取数据
    trainX, trainY, testX, testY = load_data()

    input_size = trainX.shape[1]  # 输入层大小
    output_size = trainY.shape[1]  # 输出层大小
    hidden_size = int(size)  # 隐藏层大小
    learning_rate = eta  # 学习速率
    loop_size = int(max_loop)  # 训练次数

    # 定义一个 输入神经元为input_size、隐藏层神经元为hidden_size、输出神经元为output_size 的bp神经网络
    net = Net(input_size, hidden_size, output_size)
    # print("神经网络模型：", net)

    # 定义优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.ASGD(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Rprop(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adagrad(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adadelta(net.parameters(), lr=learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)

    # 定义损失函数
    loss_func = torch.nn.L1Loss()  # MAE 平均绝对误差
    # loss_func = torch.nn.MSELoss() # MSE 均方误差

    # 进行训练
    for step in range(loop_size):
        prediction = net(trainX)
        loss = loss_func(prediction, trainY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # if step % (loop_size / 10) == 0:
        #     print("第 %d 次训练后, 训练集损失函数为：%g" % (step, loss.item()))
        if (loss.item() < threshold):
            print("第 %d 次训练后, 训练集损失函数为：%g" % (step, loss.item()), ", 小于设定阈值: %f" % threshold)
            break

    # 使用训练后的神经网络进行预测
    prediction = net(testX)

    # 计算训练结果的误差
    loss = loss_func(prediction, testY)
    print("训练结束, 当前神经网络的隐藏层层数为%d, 学习率设定为%f, 损失阈值设定为%f, 经过 %d 次训练后, 测试集损失函数为：%g" % (
        hidden_size, learning_rate, threshold, step, loss.item()))

    # 对真实值和预测值进行反归一化处理，用于画图
    realValue = train_labels_scaler.inverse_transform(testY.numpy())
    result = test_labels_scaler.inverse_transform(prediction.detach().numpy())
    draw(realValue, result)

    return loss.item()


# 画图工具，画出真实值和预测值对应图
def draw(realValue, result):
    # print(realValue)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.plot(realValue, result, 'ro')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(0, 3000)
    plt.ylim(0, 3000)
    plt.show()


# 进行多次训练，画出训练次数和错误率的折线图
if __name__ == "__main__":
    train_times = 5  # 训练次数

    X = []
    Y = []
    for i in range(0, train_times):
        print("第%d次训练：" % i)
        X.append(i)
        Y.append(aim_function(21, 0.02, 0.07))
    plt.plot(X, Y, 'r')
    plt.show()
