import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as hc

data=pd.read_csv('mouse_cluster.csv')
print(data.head(5))

#系统聚类
z=hc.linkage(data.iloc[:,1:],method='ward')#去掉braand列 'ward'：离差平方和最小法
print(z)
hc.dendrogram(z,orientation='right',labels=list(data.iloc[:,0]))
# plt.show()

#聚类结果
cluster_result=hc.fcluster(z,2.6,criterion='distance')
print(cluster_result)

#KMeans
from scipy.cluster.vq import kmeans2

result=kmeans2(data.iloc[:,1:],3)
print(result)

from sklearn.cluster import KMeans

clf = KMeans(n_clusters=3)