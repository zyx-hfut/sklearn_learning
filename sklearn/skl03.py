
# 导入拆分数据集
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# 导入make_regression
from sklearn.datasets import make_regression
# 自动生成数据 （默认是线性的）
x, y = make_regression(n_samples=100, n_features=2, random_state=10)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10, test_size=0.2)
# 创建模型
model = LinearRegression()
# 拟合
model.fit(x_train, y_train)

print('train score:', model.score(x_train, y_train))
print('test score :', model.score(x_test, y_test))
