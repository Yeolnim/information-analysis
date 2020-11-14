import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('/iris/iris.csv')
print(data.head(5))
# print(data.iloc[:,:4])

from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(data.iloc[:,0:4])#iloc:按特定的索引号 [行,列]
print(pca.explained_variance_ratio_)#贡献率

newdata=pca.fit_transform(data.iloc[:,0:4])
print(newdata)

plt.scatter(newdata[0:50,0],newdata[0:50,1],c='r')
plt.scatter(newdata[50:100,0],newdata[50:100,1],c='b')
plt.scatter(newdata[100:150,0],newdata[100:150,1],c='y')
plt.show()
