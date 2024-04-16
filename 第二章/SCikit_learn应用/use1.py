import matplotlib
import numpy as np

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error, r2_score
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
diabetes_X = diabetes_X[:, np.newaxis, 2]
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
diabetes_y_pred = regr.predict(diabetes_X_test)
print('计算回归系数：\n', regr.coef_)
print('计算均方差：%.2f' % mean_squared_error(diabetes_y_test, diabetes_y_pred))
print('计算决定系数：%.2f' % r2_score(diabetes_y_test, diabetes_y_pred))
#绘图
plt.scatter(diabetes_X_test, diabetes_y_test, color='black')
plt.plot(diabetes_X_test, diabetes_y_pred, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.show()
