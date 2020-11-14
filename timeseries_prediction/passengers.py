import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox

def arima_test():
    discfile = 'international_airline_passengers.csv'
    forecastnum = 5
    data = pd.read_csv(discfile, index_col=u'time')

    fig = plt.figure(figsize=(8, 6))
    # 第一幅自相关图
    ax1 = plt.subplot(411)
    fig = plot_acf(data, ax=ax1)

    # 平稳性检测
    print(u'原始序列的ADF检验结果为：', ADF(data[u'passengers']))
    # 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore

    # 差分后的结果
    D_data = data.diff().dropna()
    D_data.columns = [u'passengers差分']
    # 时序图
    D_data.plot()
    plt.show()
    # 第二幅自相关图
    fig = plt.figure(figsize=(8, 6))
    ax2 = plt.subplot(412)
    fig = plot_acf(D_data, ax=ax2)
    # 偏自相关图
    ax3 = plt.subplot(414)
    fig = plot_pacf(D_data, ax=ax3)
    plt.show()
    fig.clf()

    print(u'差分序列的ADF检验结果为：', ADF(D_data[u'passengers差分']))  # 平稳性检测

    # 白噪声检验
    print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(D_data, lags=1))  # 返回统计量和p值
    data[u'passengers'] = data[u'passengers'].astype(float)
    # 定阶
    pmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    qmax = int(len(D_data) / 10)  # 一般阶数不超过length/10
    bic_matrix = []  # bic矩阵 贝叶斯似然值
    data.dropna(inplace=True)

    # 存在部分报错，所以用try来跳过报错；存在warning，暂未解决使用warnings跳过
    data = np.array(data,dtype=np.float)
#     print('*'*20,ARIMA(data, (1, 1, 1)).fit().bic)
    import warnings
    warnings.filterwarnings('error')
    for p in range(pmax + 1):
        tmp = []
        for q in range(qmax + 1):
            try:
                tmp.append(ARIMA(data, (p, 1, q)).fit().bic)#一阶差分
            except:
                tmp.append(None)
        bic_matrix.append(tmp)
    # 从中可以找出最小值
    bic_matrix = pd.DataFrame(bic_matrix)
    print(bic_matrix)
    # 用stack展平，然后用idxmin找出最小值位置。
    p, q = bic_matrix.stack().idxmin()
    print(u'BIC最小的p值和q值为：%s、%s' % (p, q))
#     model = ARIMA(data, np.array(p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
    model = ARIMA(data, (p, 1, q)).fit()  # 建立ARIMA(0, 1, 1)模型
    model.summary2()  # 给出一份模型报告
    model.forecast(forecastnum)  # 作为期5天的预测，返回预测结果、标准误差、置信区间。

if __name__ == "__main__":
    arima_test()
    pass
