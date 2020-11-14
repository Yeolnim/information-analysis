# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame,Series

#读取文件
datafile = u'iris.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
data = pd.read_csv(datafile)
#data.drop(['category'],axis = 1,inplace=True)
featurename = examDf.columns

# 绘制散点图
p = plt.figure(figsize=(25,25)) ##设置画布
plt.title('iris')
i = -1
for f1 in featurename:
    i = i+1
    j = -1
    for f2 in featurename:
        j = j+1
        p.add_subplot(5,5,(i*5)+(j+1))
        # 设定坐标轴
        plt.xlabel(f1)
        plt.ylabel(f2)
        x = data[f1].values
        y = data[f2].values   #   Sepal.Width
        plt.scatter(x[:50], y[:50], c = 'red', s= 16) # s 是点的半径，c是颜色。
        plt.scatter(x[50:100], y[50:100], c = 'b', s= 16)
        plt.scatter(x[100:], y[100:], c = 'y', s= 16)    
plt.show()

rDf = examDf.corr()#查看数据间的相关系数
print(rDf)

# # 构建训练集，测试集

from sklearn.model_selection import train_test_split
#拆分训练集和测试集（train_test_split是存在与sklearn中的函数）
X_train,X_test,Y_train,Y_test = train_test_split(examDf['Petal.Length'],examDf['Petal.Width'],train_size=0.8,test_size=0.2)
#train为训练数据,test为测试数据,examDf为源数据,train_size 规定了训练数据的占比
X_train,X_test,Y_train,Y_test = X_train.values.reshape(-1,1),X_test.values.reshape(-1,1),Y_train.values.reshape(-1,1),Y_test.values.reshape(-1,1)
print("自变量---源数据:",examDf['Petal.Length'].shape, "；  训练集:",X_train.shape, "；  测试集:",X_test.shape)
print("因变量---源数据:",examDf['Petal.Width'].shape, "；  训练集:",Y_train.shape, "；  测试集:",Y_test.shape)
 
#散点图
plt.scatter(X_train, Y_train, color="darkgreen", label="train data")#训练集为深绿色点
plt.scatter(X_test, Y_test, color="red", label="test data")#测试集为红色点
 
#添加标签
plt.legend(loc=2)#图标位于左上角，即第2象限，类似的，1为右上角，3为左下角，4为右下角
plt.xlabel("The Connection amount of the average account")#添加 X 轴名称
plt.ylabel("The ratio of average return amount")#添加 Y 轴名称
plt.show()#显示散点图

# # 回归模型训练

from sklearn.linear_model import LinearRegression

#模型构建
model = LinearRegression()
 
#线性回归训练
model.fit(X_train,Y_train)#调用线性回归包
 
a  = model.intercept_#截距
b = model.coef_#回归系数
 
#训练数据的预测值
y_train_pred = model.predict(X_train)
#绘制最佳拟合线：标签用的是训练数据的预测值y_train_pred
plt.plot(X_train, y_train_pred, color='blue', linewidth=2, label="best line")
 
#测试数据散点图
plt.scatter(X_train, Y_train, color='darkgreen', label="train data")
plt.scatter(X_test, Y_test, color='red', label="test data")
 
#添加图标标签
plt.legend(loc=2)#图标位于左上角，即第2象限，类似的，1为右上角，3为左下角，4为右下角
plt.xlabel("The Connection amount of the average account")#添加 X 轴名称
plt.ylabel("The ratio of average return amount")#添加 Y 轴名称
plt.show()#显示图像
 
print("拟合参数:截距",a,",回归系数：",b)
print("最佳拟合线: Y = ",np.round(a,2),"+",np.round(b[0],2),"* X")#显示线性方程，并限制参数的小数位为两位
