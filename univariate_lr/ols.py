import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr

x = [3.4, 1.8, 4.6, 2.3, 3.1, 5.5, 0.7, 3.0, 2.6, 4.3, 2.1, 1.1, 6.1, 4.8, 3.8]
y = [26.2, 17.8, 31.3, 23.1, 27.5, 36.0, 14.1, 22.3, 19.6, 31.3, 24.0, 17.3, 43.2, 36.4, 26.1]

pccs=pearsonr(x,y)
spear=spearmanr(x,y)
print(pccs)
print(spear)

plt.scatter(x,y)
# plt.show()

from statsmodels.formula.api import ols#线性回归
import pandas as pd

data=pd.DataFrame({'x':x,'y':y})
model=ols('y~x',data).fit()#拟合
predicted=model.predict()#预测
print(predicted)

data=pd.read_csv('univariate_lr.csv')
x=data['square_feet']
y=data['price']

pccs=pearsonr(x,y)
spear=spearmanr(x,y)
print(pccs)
print(spear)

plt.scatter(x,y)

model=ols('y~x',data).fit()
predicted=model.predict()#预测
print(predicted)
plt.plot(x,predicted)#划线
plt.show()
