
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

"""print(type(data))
print(type(data.data))
print(type(data.target))
"""

# Load the diabetes dataset
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Split the targets into training/testing sets
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# Create linear regression object # 线性回归 准确度较差 对于非线性回归需要别的方法
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)

# The coefficients 相关系数
print("Coefficients: \n", regr.coef_)

# 截距
print(regr.intercept_)
# y=coe*x+intercept

# The mean squared error 误差 越小越好
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction r方 越接近1越好
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# 在训练集得分高，在测试集上得分低叫过拟合
# 在训练集得分低，在测试集得分高叫欠拟合
print('model score(训练集):',regr.score(diabetes_X_train,diabetes_y_train))
print('model score(测试集):',regr.score(diabetes_X_test,diabetes_y_test))


# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()