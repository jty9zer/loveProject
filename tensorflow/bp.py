import math

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

import warnings

warnings.filterwarnings('ignore')


# 导入数据集，本文用的是mnist手写数据集，该数据主要是对手写体进行识别0-9的数字
def load_data():
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
    trainX = features[:100]
    trainY = labels[:100]
    testX = features[100:]
    testY = labels[100:]

    return trainX, trainY, testX, testY


# 这里定义的即是评价lstm效果的函数——也是遗传算法的适应度函数
def aim_function(q, eta):
    x_train, y_train, x_test, y_test = load_data()
    d = 10  # 输入节点个数
    l = 1  # 输出节点个数
    q = math.ceil(q)  # 隐层个数,采用经验公式2d+1
    # eta = num[1]  # 学习率
    error = 0.10

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
    train_step = tf.train.GradientDescentOptimizer(eta).minimize(mse)  # 梯度下降法

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()  # 初始化节点
        sess.run(init_op)

        STEPS = 0
        while True:
            sess.run(train_step, feed_dict={x: x_train, y_: y_train})
            STEPS += 1
            train_mse = sess.run(mse, feed_dict={x: x_train, y_: y_train})
            if STEPS % 10 == 0:  # 每训练100次，输出损失函数
                print("第 %d 次训练后,训练集损失函数为：%g" % (STEPS, train_mse))
            if train_mse < error:
                break
            if STEPS == 1000:
                break
        print("总训练次数：", STEPS)

        # 测试
        Normal_y = sess.run(y, feed_dict={x: x_test})  # 求得测试集下的y计算值
        test_mse = sess.run(mse, feed_dict={y: Normal_y, y_: y_test})  # 计算误差
        print("测试集误差为：", test_mse)
        return test_mse

