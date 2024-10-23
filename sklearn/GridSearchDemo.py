from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
iris = datasets.load_iris()
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters)
print(clf.fit(iris.data, iris.target))

# 分数
print(clf.score(iris.data,iris.target))
# 最佳参数
print(clf.best_params_)

print(sorted(clf.cv_results_.keys()))

from sklearn.linear_model import Lasso
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
x,y = load_diabetes(return_X_y=True)
# 拆分数据集
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.21,random_state=10)
# 构建模型
p={'alpha':[100,1,0.1,0.001,0.0001],'max_iter':[1000,2000]}
model=Lasso()
clf = GridSearchCV(model,p)
clf.fit(x,y)
print(clf.best_params_)
print(clf.best_score_)