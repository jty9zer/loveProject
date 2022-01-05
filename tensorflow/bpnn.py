import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')


# 读取数据
data = pd.read_csv("../train_data.csv", sep=',', header=0)

features_df = data.loc[:, ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10"]]
label_df = data.loc[:, ["Y"]]

# 归一化数据
feature_scaler = MinMaxScaler(feature_range=(0, 1))
labels_scaler = MinMaxScaler(feature_range=(0, 1))
features = feature_scaler.fit_transform(features_df)
labels = labels_scaler.fit_transform(label_df)

# 截取训练数据
X = features[:98]
Y = labels[:98]
testX = features[98:]
testY = labels[98:]


# 定义神经网络的参数
d = 10  # 输入节点个数
l = 1  # 输出节点个数
# q = 2 * d + 1  # 隐层个数,采用经验公式2d+1
q = 11
# train_num = 50  # 训练数据个数
# test_num = 5  # 测试数据个数
eta = 0.5  # 学习率
error = 0.10  # 精度

# 初始化权值和阈值
w1 = tf.Variable(tf.random_normal([d, q], stddev=1, seed=1))  # seed设定随机种子，保证每次初始化相同数据
b1 = tf.Variable(tf.constant(0.0, shape=[q]))
w2 = tf.Variable(tf.random_normal([q, l], stddev=1, seed=1))
b2 = tf.Variable(tf.constant(0.0, shape=[l]))

# 输入占位
x = tf.placeholder(tf.float32, shape=(None, d))  # 列数是d，行数不定
y_ = tf.placeholder(tf.float32, shape=(None, l))

# 构建图：前向传播
a = tf.nn.sigmoid(tf.matmul(x, w1) + b1)  # sigmoid激活函数
y = tf.nn.sigmoid(tf.matmul(a, w2) + b2)
maes = tf.losses.absolute_difference(y_, y)
mse = tf.reduce_mean(maes)
# train_step = tf.train.AdamOptimizer(eta).minimize(mse)  # Adam算法
train_step = tf.train.GradientDescentOptimizer(eta).minimize(mse)  # 梯度下降法

# 创建会话来执行图
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()  # 初始化节点
    sess.run(init_op)

    STEPS = 0
    while True:
        sess.run(train_step, feed_dict={x: X, y_: Y})
        STEPS += 1
        train_mse = sess.run(mse, feed_dict={x: X, y_: Y})
        if STEPS % 10 == 0:  # 每训练100次，输出损失函数
            print("第 %d 次训练后,训练集损失函数为：%g" % (STEPS, train_mse))
        if train_mse < error:
            break
        if STEPS == 1000:
            break
    print("总训练次数：", STEPS)

    # 测试
    Normal_y = sess.run(y, feed_dict={x: testX})  # 求得测试集下的y计算值
    test_mse = sess.run(mse, feed_dict={y: Normal_y, y_: testY})  # 计算误差
    print("测试集误差为：", test_mse)


    def doPredict(nparray, label):
        normal_features = feature_scaler.transform(nparray.reshape(1, -1))  # 归一化
        XX = tf.constant(normal_features)
        a = tf.nn.sigmoid(tf.matmul(XX, w1) + b1)
        y = tf.nn.sigmoid(tf.matmul(a, w2) + b2)
        normal_result = sess.run(y)
        result = labels_scaler.inverse_transform(normal_result.reshape(1, -1))  # 反归一化
        print("实际值为[", label, "], 预测值为：", result, ", 相差： ", (result - label) / label)


    doPredict(np.array([28, 18, 0, 1986.74, 1657.55, 863.13, 1413.33, 874.96, 796.48, 896.47]).astype(np.float32),
              1038.65)
    doPredict(np.array([25, 17, 0, 1665.41, 1986.74, 1657.55, 863.13, 1413.33, 874.96, 796.48]).astype(np.float32),
              896.47)
    doPredict(np.array([31, 21, 0, 1337.74, 1665.41, 1986.74, 1657.55, 863.13, 1413.33, 874.96]).astype(np.float32),
              796.48)
    doPredict(np.array([26, 21, 0, 957.32, 1337.74, 1665.41, 1986.74, 1657.55, 863.13, 1413.33]).astype(np.float32),
              874.96)
    doPredict(np.array([34, 22, 1, 884.13, 957.32, 1337.74, 1665.41, 1986.74, 1657.55, 863.13]).astype(np.float32),
              1413.33)
    doPredict(np.array([32, 23, 1, 908.17, 884.13, 957.32, 1337.74, 1665.41, 1986.74, 1657.55]).astype(np.float32),
              863.13)
    doPredict(np.array([33, 23, 0, 1767.36, 908.17, 884.13, 957.32, 1337.74, 1665.41, 1986.74]).astype(np.float32),
              1657.55)
    doPredict(np.array([35, 21, 0, 1463.57, 1767.36, 908.17, 884.13, 957.32, 1337.74, 1665.41]).astype(np.float32),
              1986.74)
    doPredict(np.array([35, 21, 0, 1362.37, 1463.57, 1767.36, 908.17, 884.13, 957.32, 1337.74]).astype(np.float32),
              1665.41)
    doPredict(np.array([33, 19, 0, 1402.26, 1362.37, 1463.57, 1767.36, 908.17, 884.13, 957.32]).astype(np.float32),
              1337.74)
    doPredict(np.array([26, 19, 0, 1409.23, 1402.26, 1362.37, 1463.57, 1767.36, 908.17, 884.13]).astype(np.float32),
              957.32)
    doPredict(np.array([34, 18, 1, 462.42, 1409.23, 1402.26, 1362.37, 1463.57, 1767.36, 908.17]).astype(np.float32),
              884.13)
    doPredict(np.array([32, 21, 1, 684.08, 462.42, 1409.23, 1402.26, 1362.37, 1463.57, 1767.36]).astype(np.float32),
              908.17)
    doPredict(np.array([31, 20, 0, 1755.12, 684.08, 462.42, 1409.23, 1402.26, 1362.37, 1463.57]).astype(np.float32),
              1767.36)
    doPredict(np.array([30, 17, 0, 1354.4, 1755.12, 684.08, 462.42, 1409.23, 1402.26, 1362.37]).astype(np.float32),
              1463.57)
