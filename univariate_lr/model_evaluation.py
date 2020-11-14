from statsmodels.formula.api import ols#线性回归
import pandas as pd

x = [3.4, 1.8, 4.6, 2.3, 3.1, 5.5, 0.7, 3.0, 2.6, 4.3, 2.1, 1.1, 6.1, 4.8, 3.8]
y = [26.2, 17.8, 31.3, 23.1, 27.5, 36.0, 14.1, 22.3, 19.6, 31.3, 24.0, 17.3, 43.2, 36.4, 26.1]

data=pd.DataFrame({'x':x,'y':y})
model=ols('y~x',data).fit()#拟合
#调整维度 series类型
x_in=data['x'].values.reshape(-1,1)#-1自动算维度
y_in=data['y'].values.reshape(-1,1)
#预测 series类型
predicted=model.predict()
print(model.summary())
print(predicted)

from statsmodels.stats.anova import anova_lm
anovat=anova_lm(model)#方差分析
print(anovat)

import numpy as np
def Residual_plot(x,y,y_prd):
    # from matplotlib.font_manager import FontProperties

    n=len(x)
    e=y-y_prd
    sigama=np.std(e)#残差的标准差

    #绘图
    import matplotlib.pyplot as plt
    mx=max(x)[0]+1
    plt.scatter(x,e,c='red',s=6)
    plt.plot([0,mx],[2*sigama,2*sigama],'k--',c='green')
    plt.plot([0, mx], [-2 * sigama, -2 * sigama], 'k--', c='green')
    plt.plot([0, mx], [3 * sigama, 3 * sigama], 'k--', c='orange')
    plt.plot([0, mx], [-3 * sigama, -3 * sigama], 'k--', c='orange')
    plt.xlim(0,mx)
    plt.ylim(-np.ceil(3*sigama+2),np.ceil(3*sigama+2))
    plt.show()

Residual_plot(x_in,y_in,predicted.reshape(-1,1))
