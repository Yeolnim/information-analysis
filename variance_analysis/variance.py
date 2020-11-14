from scipy.stats import f
import numpy as np

def main():
    a = [[81, 82, 79, 81, 78, 89, 92, 87, 85, 86],
         [71, 72, 72, 66, 72, 77, 81, 77, 73, 79],
         [76, 79, 77, 76, 78, 89, 87, 84, 87, 87]]
    alpha=0.05
    group_num=len(a)
    ingroup_num=len(a[0])
    group_mean=[np.mean(ele) for ele in a]
    print("group_mean=",group_mean)
    grand_mean=np.mean(group_mean)
    print("grand_mean=",grand_mean)
    #SSA处理效应
    SSA=sum([ingroup_num*((ele-grand_mean)**2) for ele in group_mean])
    #MSA
    MSA=SSA/(group_num-1)

    print('df1=',group_num-1)
    print('SSA=',SSA)
    print('MSA=',MSA)

    SSE=0
    for i in range(group_num):
        for j in range(ingroup_num):
            SSE+=(a[i][j]-group_mean[i])**2

    MSE=SSE/(group_num*(ingroup_num-1))
    print(group_num*(ingroup_num-1))
    print('MSE=',MSE)
    F_practical=MSA/MSE
    print('F_practical=',F_practical)
    F_expected=f.ppf(1-alpha,group_num-1,group_num*(ingroup_num-1))
    print('F_expected=',F_expected)
    print('df2',group_num*(ingroup_num-1))
    #计算P
    P=f.cdf(F_practical,group_num-1,group_num*(ingroup_num-1))
    print('p',P)
    if F_practical>F_expected:
        return False
    return True

if __name__ == '__main__':
    print(main())
