# coding: utf-8

import seaborn as sns  # 可以画置信区间的图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#读取文件
datafile = u'/iris/iris.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
examDf = pd.read_csv(datafile)
examDf.drop(['category'],axis = 1,inplace=True)
featurename = examDf.columns
print(featurename)
sns.pairplot(examDf, x_vars=('Sepal.Length', 'Sepal.Width','Petal.Width'), y_vars=('Petal.Length'), size=7, aspect=0.8, kind='reg')  
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
#拆分训练集和测试集
X_train,X_test,Y_train,Y_test = train_test_split(examDf.ix[:,[0,3]],examDf['Petal.Length'],train_size=0.8,test_size=0.2)
#new_examDf.ix[:,:2]取了数据中的前两列为自变量，此处与单变量的不同
#X_train,X_test,Y_train,Y_test = X_train.values.reshape(-1,1),X_test.values.reshape(-1,1),Y_train.values.reshape(-1,1),Y_test.values.reshape(-1,1)
 
print("自变量---源数据:",examDf.ix[:,[0,3]].shape, "；  训练集:",X_train.shape, "；  测试集:",X_test.shape)
print("因变量---源数据:",examDf['Petal.Length'].shape, "；  训练集:",Y_train.shape, "；  测试集:",Y_test.shape)
 
#调用线性规划包
model = LinearRegression()
 
model.fit(X_train,Y_train)#线性回归训练
 
a  = model.intercept_#截距
b = model.coef_#回归系数
print("拟合参数:截距",a,",回归系数：",b)
 
#显示线性方程，并限制参数的小数位为两位
print("最佳拟合线: Y = ",round(a,2),"+",round(b[0],2),"* X1 + ",round(b[1],2),"* X2")
 
Y_pred = model.predict(X_test)#对测试集数据，用predict函数预测
 
plt.plot(range(len(Y_pred)),Y_pred,'red', linewidth=2.5,label="predict data")
plt.plot(range(len(Y_test)),Y_test,'green',label="test data")
plt.legend(loc=2)
plt.show()
