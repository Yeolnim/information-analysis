import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('live.csv',encoding='gb2312')
print(data.head(5))

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(data.iloc[:,1:8])#iloc:按特定的索引号 [行,列]
print(pca.explained_variance_ratio_)#贡献率

newdata=pca.fit_transform(data.iloc[:,1:8])
print(newdata)

plt.scatter(newdata[:,0],newdata[:,1])
plt.show()

#标准化：减去均值除以标准差

#法2
from matplotlib.mlab import PCA as mlabPCA

live_pcl=mlabPCA(data.iloc[:,1:8],standardize=True)
live_eigenvector=pd.DataFrame(live_pcl.Wt,index=['P1','P2','P3','P4','P5','P6','P7'],columns=data.columns[1:8])#转成df,设定索引
live_eigenvector=live_eigenvector.T
print(live_eigenvector)