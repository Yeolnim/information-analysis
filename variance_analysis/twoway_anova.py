import pandas as pd
df=pd.read_excel('双因子方差分析.xlsx')
print(df)

#计量经济学模型
from statsmodels.formula.api import ols#线性回归
from statsmodels.stats.anova import anova_lm#方差分析,线性模型，y：因变量

#主效应分析
formula='销售额~C(竞争者数量)+C(超市位置)'
model=ols(formula,df).fit()#线性拟合
anovat=anova_lm(model)#方差分析
print(anovat)

#交互效应分析
formula='销售额~C(竞争者数量)+C(超市位置)+C(竞争者数量):C(超市位置)'
model=ols(formula,df).fit()#线性拟合
anovat=anova_lm(model)#方差分析
print(anovat)
