
# Non-Negative Least Squares 非负最小二乘法
# 对回归系数具有正约束性
# LinearRegression 的positive 设为True即可
from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
# return_X_Y=False 返回Bunch =True 返回data和target两部分 方便处理
# data = datasets.load_diabetes(return_X_y=False)
#
# print(data)
# Description
# print(data.DESCR)
# 具体要看API文档
#
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 加载数据集
# x,y = load_boston(return_X_y=True)
x, y = load_diabetes(return_X_y=True)
# 拆分数据集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=10)
# 构建模型
model = LinearRegression(positive=True)  # 非负数设定
# model = LinearRegression()
# 训练数据
model.fit(x_train, y_train)
print(model.coef_)
# 验证模型是否存在欠拟合/过拟合
print('train score:', model.score(x_train, y_train))
print('test score:', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print(mean_squared_error(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
