# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文
plt.rcParams['axes.unicode_minus']=False     # 正常显示负号
# 随机生成（10000,）服从正态分布的数据
#读取文件
datafile = u'D:\project/10信息分析方法与工具\一元线性回归\iris\iris.csv'#文件所在位置，u为防止路径中有中文名称，此处没有，可以省略
examDf = pd.read_csv(datafile)
examDf.drop(['category'],axis = 1,inplace=True)
featurename = examDf.columns
data = examDf['Sepal.Length']
#data = np.random.randn(10000)
"""
绘制直方图
data:必选参数，绘图数据
bins:直方图的长条形数目，可选项，默认为10
normed:是否将得到的直方图向量归一化，可选项，默认为0，代表不归一化，显示频数。normed=1，表示归一化，显示频率。
facecolor:长条形的颜色
edgecolor:长条形边框的颜色
alpha:透明度
"""
for fn in featurename:
    data = examDf[fn]
    plt.hist(data, bins=10, normed=0, facecolor="blue", edgecolor="black", alpha=0.7)
    # 显示横轴标签
    plt.xlabel("区间")
    # 显示纵轴标签
    plt.ylabel(fn+" 频数/频率")
    # 显示图标题
    plt.title("频数/频率分布直方图")
    plt.show()

#饼图
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm #字体管理器

#准备字体
#my_font = fm.FontProperties(fname="/usr/share/fonts/wqy-microhei/wqy-microhei.ttc")
#准备数据
p = plt.figure(figsize=(4,4)) ##设置画布
plt.title('2018年8月的编程语言指数排行榜')#,fontproperties=my_font

data = [0.16881,0.14966,0.07471,0.06992,0.04762,0.03541,0.02925,0.02411,0.02316,0.01409,0.36326]
#准备标签
labels = ['Java','C','C++','Python','Visual Basic.NET','C#','PHP','JavaScript','SQL','Assembly langugage','其他']
#将排列在第4位的语言(Python)分离出来
explode =[0,0,0,0.3,0,0,0,0,0,0,0]
#使用自定义颜色

colors = ['red','pink','magenta','purple','orange']

#将横、纵坐标轴标准化处理,保证饼图是一个正圆,否则为椭圆
plt.axes(aspect='equal')

#控制X轴和Y轴的范围(用于控制饼图的圆心、半径)
plt.xlim(0,8)
plt.ylim(0,8)

#不显示边框
'''
plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['left'].set_color('none')
plt.gca().spines['bottom'].set_color('none')
'''

#绘制饼图

plt.pie(x=data, #绘制数据
labels=labels,#添加编程语言标签
explode=explode,#突出显示Python
colors=colors, #设置自定义填充色
autopct='%.3f%%',#设置百分比的格式,保留3位小数
pctdistance=0.8, #设置百分比标签和圆心的距离
labeldistance=1.2,#设置标签和圆心的距离
startangle=180,#设置饼图的初始角度
center=(4,4),#设置饼图的圆心(相当于X轴和Y轴的范围)
radius=3.8,#设置饼图的半径(相当于X轴和Y轴的范围)
counterclock= False,#是否为逆时针方向,False表示顺时针方向
wedgeprops= {'linewidth':1,'edgecolor':'green'},#设置饼图内外边界的属性值
textprops= {'fontsize':12,'color':'black'},#设置文本标签的属性值
frame=0) #是否显示饼图的圆圈,1为显示

#不显示X轴、Y轴的刻度值
plt.xticks(())
plt.yticks(())

plt.show()
