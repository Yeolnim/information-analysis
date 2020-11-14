#一元线性回归模型
import scipy.stats as sst
import numpy as np
import matplotlib.pylab as plt
from matplotlib.widgets import Cursor
import csv


#对升序序列找到对匹配val值最接近的点坐标
def index_fit(y, val):
    if val >= y[-1]:
        return len(y) - 1
    for i, yi in enumerate(y):
        if val >= yi and val <= y[i + 1]:
            if abs(val - yi) <= abs(val - y[ i + 1]):
                fit_index = i
            else:
                fit_index = i+1
            break
    return fit_index

#用于计算Lxx, Lyy
def laa(x):
    x_mean = np.mean(x)
    lxx = np.sum((x-x_mean)**2)
    return lxx

#用于计算Lxy
def lab(x,y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    lxy = np.sum((x-x_mean)*(y-y_mean))
    return lxy


#一元线性回归模型
def polyfit_one(x, y, alpha):
    assert len(x) == len(y)
    n = len(x)
    assert n > 2
    lxx = laa(x)
    lyy = laa(y)
    lxy = lab(x, y)

    R = lxy/(np.sqrt(lxx) * np.sqrt(lyy))
    R2 = R*R   #计算相关系数与决定系数

    b_est = lxy/lxx  #计算b估计
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    a_est = y_mean - b_est * x_mean   #计算a估计
    Qe = lyy - b_est * lxy
    sigma_est2 = Qe / (n - 2)

    sigma_est = np.sqrt(sigma_est2) #sigma估计

    test = np.abs(b_est * np.sqrt(lxx))/sigma_est
    test_level = sst.t.ppf(1 - alpha/2, df=n - 2)
    linear_test = test > test_level   #线性回归检验

    #a,b的置信区间
    b_int = [b_est - test_level * sigma_est / np.sqrt(lxx), b_est + test_level * sigma_est / np.sqrt(lxx)]
    a_int = [y_mean - b_int[1] * x_mean, y_mean - b_int[0] * x_mean]

    poly_int = (a_int, b_int)

    poly_val = (a_est, b_est)

    #返回回归模型相应参数
    test_val = {'R': R,
                'R2': R2,
                'linear_test': linear_test,
                'poly_int': poly_int,
                }
    process_val = {'lxx': lxx,
                   'lyy': lyy,
                   'lxy': lxy,
                   'sigma_est': sigma_est,
                   'x_mean': x_mean,
                   'y_mean': y_mean,
                   'test_level': test_level,
                   'ndim': n,
                   }
    return (poly_val, test_val, process_val)

#计算相应的预测区间
def confidence_interval(y0=None, *args, **kwargs):
    a_est, b_est = args
    sigma_est = kwargs['sigma_est']
    test_level= kwargs['test_level']
    lxx = kwargs['lxx']
    n = kwargs['ndim']
    x_mean = kwargs['x_mean']

    if isinstance(y0, (int, float, np.ndarray)):
        x0 = (y0 - a_est) / b_est
    elif isinstance(y0, (list, tuple)):
        y0 = np.array(y0)
        x0 = (y0 - a_est) / b_est
    else:
        return None

    conf_down = y0 - test_level * sigma_est * np.sqrt(1 + 1 / n + ((x0 - x_mean) ** 2 / lxx))
    conf_up = y0 + test_level * sigma_est * np.sqrt(1 + 1 / n + ((x0 - x_mean) ** 2 / lxx))

    confidence_interval = (conf_down, conf_up)

    return confidence_interval

# 构建回归模型图示，
# #parama:alpha 置信水平 ，fignum表示绘画窗口号，
#         kwargs 设置ylimit line:ylimit_down, ylimit_up , axis tick: ytick_down, ytick_up
#         xlabel, ylabel, lengend label, xtick, ytick,title
def figure_drawing(x,y,alpha, fig_num, **kwargs):
    poly_val, test_val, process_val = polyfit_one(x, y, alpha)
    down_zone = confidence_interval(4500, *poly_val, **process_val)
    up_zone = confidence_interval(7000, *poly_val, **process_val)

    print(poly_val)
    print(test_val)
    print(process_val)
    print(down_zone, up_zone)
    print("Linear Test:", test_val['linear_test'])

    R2 = test_val['R2']
    f = plt.figure(fig_num)
    # seaborn.set()

    ax = f.add_subplot(111)

    ylimit_down = kwargs['ylimit_down']
    ylimit_up = kwargs['ylimit_up']

    tick_yd = kwargs['ytick_down']  # int(ylimit_down * 0.7)
    tick_yu = kwargs['ytick_up']  # int(ylimit_up * 1.3)
    Y_test = np.linspace(tick_yd, tick_yu, 1000) #从ticks上下限间取1000个点
    X_test = (Y_test - poly_val[0]) / poly_val[1]

    Y_down, Y_up = confidence_interval(Y_test, *poly_val, **process_val)

    xd = X_test[index_fit(Y_down, ylimit_down)]
    xu = X_test[index_fit(Y_up, ylimit_up)]
    yd = poly_val[0] + poly_val[1] * xd
    yu = poly_val[0] + poly_val[1] * xu

    xy_text1 = "(%.2f, %.2f)" % (xd, yd)
    xy_text2 = "(%.2f, %.2f)" % (xu, yu)
    poly_text = "y = %.2f *x  + %.2f " % (poly_val[1], poly_val[0])
    ax.plot(X_test, Y_test, '-b', label=poly_text)

    ax.plot(X_test, Y_down, '--g')
    ax.plot(X_test, Y_up, '--y')
    ax.scatter(x, y, s=10, c='r')
    xlabel = kwargs['xlabel']
    ylabel = kwargs['ylabel']
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    xticks = kwargs['xticks']
    yticks = kwargs['yticks']
    ax.set_yticks(yticks)
    ax.set_xticks(xticks)

    title = kwargs['title'] + "\n $R^2$ =%f" % R2
    ax.set_title(title)

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=-30)

    plt.axvline(xd, linestyle='-', color='purple', label="XL:%.2f " % xd)
    plt.axvline(xu, linestyle='-', color='cyan', label="XH:%.2f " % xu)

    legend = ax.legend(loc="upper left")
    legend_f = legend.get_frame()
    # legend_f.set_alpha(0)
    legend_f.set_facecolor("white")


    plt.annotate(xy_text1, xy=(xd, yd),
                 xycoords='data',
                 xytext=(+20, -30), textcoords='offset points',
                 fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad= -.3"))

    plt.annotate(xy_text2, xy=(xu, yu),
                 xycoords='data',
                 xytext=(+20, -30), textcoords='offset points',
                 fontsize=10, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad= -.3"))

    plt.axhline(ylimit_down, linestyle=':', color='brown')
    plt.axhline(ylimit_up, linestyle=':', color='darkblue')
    plt.grid(True)


    return poly_val, process_val

#从csv文件里面读取x,y
def read_xy(file_path):
    with open(file_path,encoding="utf-8") as fp:
        csv_reader = csv.reader(fp)
        x = []
        y = []
        for ri,row in enumerate(csv_reader):
            if ri == 0:
                continue
            x.append(float(row[0]))
            y.append(float(row[1]))
    return np.array(x), np.array(y)

def add_gausian_noise(xtest, snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(xtest**2)/len(xtest)
    npower = np.sqrt(xpower / snr)
    noise = np.random.randn(len(xtest))*npower
    # print("add noise:", noise)
    #受限于信号长度，实际噪声与理论噪声会有一定出入
    print("noise real power", np.sqrt(np.sum(noise**2/len(xtest))))
    return noise

if __name__ == "__main__":
    x, y = read_xy('data.csv') #读入数据
    y = y + add_gausian_noise(y, 40)  # 原信号本身存在噪声，可自行判断是否再加入噪声
    fig_opt = {
        'title':  'Resistance to Temperature with 99% confidence',
        'xlabel': 'Temperature /$\degree$C ',
        'ylabel': 'Resistance /m$\Omega$',
        'ylimit_down': 4500,
        'ylimit_up': 7000,
        'ytick_down': 4000,
        'ytick_up': 8000,
        'xticks': np.arange(-20, 210, 10),
        'yticks': np.arange(4000, 8000, 500),
    }
    poly_val, process_val = figure_drawing(x, y, 0.01, 1, **fig_opt)
    reserve_i = []
    # 剔除异常点
    for i, xi in enumerate(x):
        y0 = poly_val[0] + poly_val[1] * xi
        if y[i] >= confidence_interval(y0, *poly_val, **process_val)[0] and                y[i] <= confidence_interval(y0, *poly_val, **process_val)[1]:
            reserve_i.append(i)
    x = x[reserve_i]
    y = y[reserve_i]
    print(reserve_i)
    print(len(reserve_i))
    #剔除异常点后，再进行线性回归图示，此时取alpha为之前的两倍
    figure_drawing(x, y, 0.02, 2, **fig_opt)
    #添加游动光标
    cursor = Cursor(plt.gca(), horizOn=True, color='black', lw=1)
    plt.show()
