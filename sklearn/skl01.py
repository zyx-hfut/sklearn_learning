
from sklearn import linear_model as lm
import matplotlib.pyplot as plt
import numpy as np

'''
# 生成-5到5之间的100个数
x=np.linspace(-5,5,100)
# 输入直线方程
y = 0.5*x+3
# 画个直线
plt.plot(x,y)
plt.show()
'''
# 四个样本的线性模型 2个特征(x1,x2) 1个目标变量(y)
# 用fit对arrays X进行拟合
reg = lm.LinearRegression()
model = reg.fit([[0, 0], [1, 1], [2, 2], [1, 0]], [0, 1, 2, 0.5])  # fit([x],[y])
# w1*x10+w2*x20=0 # 0+0=0
# w1*x11+w2*x21=1 # w1+w2=1
# w1*x12+w2*x22=2 # 2*w1+2*w2=2
# w1*x13+w2*x23=0.5 # w1+0=0.5
# 解得w1=w2=0.5
print(model.coef_)
