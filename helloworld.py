from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
import numpy

model = Sequential()
model.add(Dense(500, input_dim=784))  # 输入层，28*28=784
model.add(Activation('sigmoid'))  # 激活函数
# model.add(Dropout(0.5))  # 采用50%的dropout

for i in range(6):
    model.add(Dense(500))
    model.add(Activation('sigmoid'))
    # model.add(Dropout(0.5))

model.add(Dense(10))  # 输出结果是10个类别，所以维度是10
model.add(Activation('softmax'))  # 最后一层用softmax

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  # 设定学习率（lr）等参数
model.compile(
    loss='categorical_crossentropy', optimizer='adam',
    metrics=['accuracy'])  # 使用交叉熵作为loss函数

(X_train,
 y_train), (X_test,
            y_test) = mnist.load_data()  # 使用Keras自带的mnist工具读取数据（第一次需要联网）

# print(X_train.shape)
# print(y_train.shape)
# exit()

# 由于mist的输入数据维度是(num, 28, 28)，这里需要把后面的维度直接拼起来变成784维
X_train = X_train.reshape(
    X_train.shape[0], X_train.shape[1] * X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2])
Y_train = (numpy.arange(10) == y_train[:, None]).astype(
    int)  # 参考上一篇文章，这里需要把index转换成一个one hot的矩阵
Y_test = (numpy.arange(10) == y_test[:, None]).astype(int)

# 开始训练，这里参数比较多。
# batch_size就是batch_size，
# epochs就是最多迭代的次数
# shuffle就是是否把数据随机打乱之后再进行训练
# verbose日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
# 就是说0是不屏显，1是显示一个进度条，2是每个epoch都显示一行数据
# validation_split就是拿出百分之多少用来做交叉验证
model.fit(
    X_train,
    Y_train,
    batch_size=2000,
    epochs=10,
    shuffle=True,
    verbose=1,
    validation_split=0.3)
print('test set')
result = model.evaluate(X_test, Y_test, batch_size=500, verbose=1)
print('loss: %s, acc: %s' % (result[0],  result[1]))
