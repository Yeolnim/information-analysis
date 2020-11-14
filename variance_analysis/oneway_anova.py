import pandas as pd

data_value={'品种1':[81, 82, 79, 81, 78, 89, 92, 87, 85, 86],
         '品种2':[71, 72, 72, 66, 72, 77, 81, 77, 73, 79],
         '品种3':[76, 79, 77, 76, 78, 89, 87, 84, 87, 87]}
# da=pd.DataFrame(data_value)
# print(da)
da=pd.DataFrame(data_value).stack()#把列变成索引
# print(da)
da=da.reset_index(level=1)#重置下标为1的索引
da.columns=['品种','产量']
print(da)

#计量经济学模型
from statsmodels.formula.api import ols#线性回归
from statsmodels.stats.anova import anova_lm#方差分析,线性模型，y：因变量

#线性回归拟合，因变量（数值型:产量）和自变量（类别变量：品种）
formula='产量~C(品种)'
model=ols(formula,da).fit()#线性拟合
anovat=anova_lm(model)#方差分析
print(anovat)
print(anovat['F'][0])#打印特定值
