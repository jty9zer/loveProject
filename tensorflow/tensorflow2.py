import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib

matplotlib.use("TkAgg")  # 这个设置可以使matplotlib保存.png图到磁盘
import matplotlib.pyplot as plt

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
X = features[:100]
Y = labels[:100]
testX = features[100:]
testY = labels[100:]

# 构建一个结构为[10,15,1]的BP神经网络
model = tf.keras.Sequential([tf.keras.layers.Dense(21, activation='relu', input_shape=(10,)),
                             tf.keras.layers.Dense(1)])
model.summary()  # 显示网络结构
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=['accuracy']
)  # 定义优化方法为随机梯度下降，损失函数为mae

# x->训练集,y——>bia标签,epochs=10000训练的次数,validation_data=(test_x,test_y)——>验证集
history = model.fit(X, Y, epochs=1000, validation_data=(testX, testY))

N = np.arange(1, 1001, 1)
plt.figure()
plt.plot(N, history.history['loss'], label='train_loss')
# plt.scatter(N, history.history['loss'])
plt.plot(N, history.history['val_loss'], label='val_loss')
# plt.scatter(N, history.history['val_loss'])
plt.title('Training Loss on Our_dataset')
plt.xlabel('Epoch #')
plt.ylabel('Loss')
plt.legend()
plt.show()
