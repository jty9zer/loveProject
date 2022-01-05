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


# feature_scaler = MinMaxScaler(feature_range=(0, 1))
# labels_scaler = MinMaxScaler(feature_range=(0, 1))

# 定义神经网络
class Net(torch.nn.Module):
    # 定义网络结构
    def __init__(self, n_input, n_hidden, n_output,
                 w1_11, w1_12, w1_13, w1_14, w1_15, w1_16,
                 w1_21, w1_22, w1_23, w1_24, w1_25, w1_26,
                 w1_31, w1_32, w1_33, w1_34, w1_35, w1_36,
                 w1_41, w1_42, w1_43, w1_44, w1_45, w1_46,
                 w1_51, w1_52, w1_53, w1_54, w1_55, w1_56,
                 w1_61, w1_62, w1_63, w1_64, w1_65, w1_66,
                 w1_71, w1_72, w1_73, w1_74, w1_75, w1_76,
                 w1_81, w1_82, w1_83, w1_84, w1_85, w1_86,
                 w1_91, w1_92, w1_93, w1_94, w1_95, w1_96,
                 w1_101, w1_102, w1_103, w1_104, w1_105, w1_106,
                 w1_111, w1_112, w1_113, w1_114, w1_115, w1_116,
                 w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9, w2_10, w2_11):
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_input, n_hidden)
        new_wight = torch.Tensor(
            np.array([[w1_11, w1_12, w1_13, w1_14, w1_15, w1_16],
                      [w1_21, w1_22, w1_23, w1_24, w1_25, w1_26],
                      [w1_31, w1_32, w1_33, w1_34, w1_35, w1_36],
                      [w1_41, w1_42, w1_43, w1_44, w1_45, w1_46],
                      [w1_51, w1_52, w1_53, w1_54, w1_55, w1_56],
                      [w1_61, w1_62, w1_63, w1_64, w1_65, w1_66],
                      [w1_71, w1_72, w1_73, w1_74, w1_75, w1_76],
                      [w1_81, w1_82, w1_83, w1_84, w1_85, w1_86],
                      [w1_91, w1_92, w1_93, w1_94, w1_95, w1_96],
                      [w1_101, w1_102, w1_103, w1_104, w1_105, w1_106],
                      [w1_111, w1_112, w1_113, w1_114, w1_115, w1_116]]
                     ).reshape((11, 6)))
        self.hidden.weight = torch.nn.Parameter(new_wight)
        # print("当前隐藏层层自定义权值", self.hidden.weight)
        self.predict = torch.nn.Linear(n_hidden, n_output)  # 此为隐藏层到输出层
        new_wight2 = torch.Tensor(
            np.array([w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9, w2_10, w2_11]).reshape((1, 11)))
        self.predict.weight = torch.nn.Parameter(new_wight2)  # 自定义权值初始化
        # print("当前输出层自定义权值", self.predict.weight)

    # 定义前向传播函数
    def forward(self, input):
        # out = self.predict(torch.sigmoid(self.hidden(input)))
        out = self.predict(torch.relu(self.hidden(input)))
        # out = self.predict(torch.tanh(self.hidden(input)))
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

    print("训练集大小：", len(trainY), "测试集大小：", len(testY))

    return trainX, trainY, testX, testY


# 目标函数，返回模型的预测误差，可用于GA遗传算法的适应度函数。
def aim_function(w1_11, w1_12, w1_13, w1_14, w1_15, w1_16,
                 w1_21, w1_22, w1_23, w1_24, w1_25, w1_26,
                 w1_31, w1_32, w1_33, w1_34, w1_35, w1_36,
                 w1_41, w1_42, w1_43, w1_44, w1_45, w1_46,
                 w1_51, w1_52, w1_53, w1_54, w1_55, w1_56,
                 w1_61, w1_62, w1_63, w1_64, w1_65, w1_66,
                 w1_71, w1_72, w1_73, w1_74, w1_75, w1_76,
                 w1_81, w1_82, w1_83, w1_84, w1_85, w1_86,
                 w1_91, w1_92, w1_93, w1_94, w1_95, w1_96,
                 w1_101, w1_102, w1_103, w1_104, w1_105, w1_106,
                 w1_111, w1_112, w1_113, w1_114, w1_115, w1_116,
                 w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9, w2_10, w2_11,
                 threshold):
    # 读取数据
    trainX, trainY, testX, testY = load_data()

    input_size = trainX.shape[1]  # 输入层大小
    output_size = trainY.shape[1]  # 输出层大小
    hidden_size = 11  # 隐藏层大小
    learning_rate = 0.0005  # 学习速率
    loop_size = 10000  # 最大训练次数

    # 定义一个 输入神经元为input_size、隐藏层神经元为hidden_size、输出神经元为output_size 的bp神经网络
    net = Net(input_size, hidden_size, output_size,
              w1_11, w1_12, w1_13, w1_14, w1_15, w1_16,
              w1_21, w1_22, w1_23, w1_24, w1_25, w1_26,
              w1_31, w1_32, w1_33, w1_34, w1_35, w1_36,
              w1_41, w1_42, w1_43, w1_44, w1_45, w1_46,
              w1_51, w1_52, w1_53, w1_54, w1_55, w1_56,
              w1_61, w1_62, w1_63, w1_64, w1_65, w1_66,
              w1_71, w1_72, w1_73, w1_74, w1_75, w1_76,
              w1_81, w1_82, w1_83, w1_84, w1_85, w1_86,
              w1_91, w1_92, w1_93, w1_94, w1_95, w1_96,
              w1_101, w1_102, w1_103, w1_104, w1_105, w1_106,
              w1_111, w1_112, w1_113, w1_114, w1_115, w1_116,
              w2_1, w2_2, w2_3, w2_4, w2_5, w2_6, w2_7, w2_8, w2_9, w2_10, w2_11)
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
    # loss_func = torch.nn.SmoothL1Loss() # 优化后的MAE
    # loss_func = torch.nn.MSELoss() # MSE 均方误差

    # 进行训练
    for step in range(loop_size):
        prediction = net(trainX)
        loss = loss_func(prediction, trainY)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % (loop_size / 10) == 0:
            print("第 %d 次训练后, 训练集损失函数为：%g" % (step, loss.item()))
        if (loss.item() < threshold):
            print("第 %d 次训练后, 训练集损失函数为：%g" % (step, loss.item()), ", 小于设定阈值: %f" % threshold)
            break

    train_loss = loss.item()
    # 使用训练后的神经网络进行预测
    prediction = net(testX)

    # 计算训练结果的误差
    loss = loss_func(prediction, testY)
    print("训练结束, 当前神经网络的隐藏层层数为%d, 学习率设定为%f, 损失阈值设定为%f, 经过 %d 次训练后, 训练集损失为：%g, 测试集损失：%g" % (
        hidden_size, learning_rate, threshold, step, train_loss, loss.item()))

    # from sklearn.metrics import mean_absolute_error  # 平方绝对误差
    # print("平均绝对误差:", mean_absolute_error(testY.numpy(),prediction.detach().numpy()))  # MAE 平均绝对误差

    # 对真实值和预测值进行反归一化处理，用于画图
    # real_Value = test_labels_scaler.inverse_transform(testY.numpy())
    # result = test_labels_scaler.inverse_transform(prediction.detach().numpy())
    # draw(real_Value, result)

    return loss.item()


# 画图工具，画出真实值和预测值对应图
def draw(real_value, result):
    # loss_fn = torch.nn.L1Loss(reduce=False, size_average=False)
    # r1 = torch.tensor(result)
    # r2 = torch.tensor(realValue)
    # loss = loss_fn(r1, r2)

    # print("真实值：", real_value, "，预测值：", result)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.xlabel('真实值')
    plt.ylabel('预测值')
    plt.plot(real_value, result, 'ro')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlim(30000, 60000)
    plt.ylim(30000, 60000)
    plt.show()


# 进行多次训练，画出训练次数和错误率的折线图
if __name__ == "__main__":
    train_times = 20  # 训练次数

    X = []
    Y = []
    for i in range(0, train_times):
        print("第%d次训练：" % i)
        X.append(i)
        Y.append(aim_function(1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                              0.10))
        # Y.append((aim_function(0.91411727, 0.61120747, -0.1390163, 0.17416743, 0.43410228, 0.80558824,
        #                        -0.5255485, -0.3918796, -0.78304105, 0.16952166, 0.79587256, -0.76037174,
        #                        0.61490701, 0.9282766, 0.31506209, 0.60384296, 0.69817763, -0.44687341,
        #                        -0.00676146, -0.75711852, -0.2669457, 0.73227989, 0.24095086, 0.8206104,
        #                        0.03507587, 0.89495802, -0.95107478, -0.70359754, 0.02001812
        #                        )))
    plt.plot(X, Y, 'r')
    plt.show()
