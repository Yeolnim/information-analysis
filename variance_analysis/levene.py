from scipy import stats

data_value={'品种1':[81, 82, 79, 81, 78, 89, 92, 87, 85, 86],
         '品种2':[71, 72, 72, 66, 72, 77, 81, 77, 73, 79],
         '品种3':[76, 79, 77, 76, 78, 89, 87, 84, 87, 87]}
args=[data_value['品种1'],data_value['品种2'],data_value['品种3']]

#方差齐性检验
w,p=stats.levene(*args)#解包，三个参数
print(w,p)

#原假设方差无差别
if p<0.05:
    print('拒绝原假设')

#单因素方差分析
f,p=stats.f_oneway(*args)
print(f,p)
