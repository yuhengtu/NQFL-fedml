import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def gaussian(t):
    return math.exp(-t**2/2)/math.sqrt(2*math.pi)

def f(t):
    return gaussian(t)

# t表示区间划分点，thresholds
# x表示区间质心，centroid/level，t的长度比x少1（两端点为正负无穷）
def interval_MSE(x,t1,t2):
    return integrate.quad(lambda t: ((t - x)**2) * f(t), t1, t2)[0]
# quad 返回一个元组 (result, error)，其中 result 是数值积分的结果，而 error 是估计的误差

def MSE(t,x):
    s = interval_MSE(x[0], -float('Inf'), t[0]) + interval_MSE(x[-1], t[-1], float('Inf'))
    for i in range(1,len(x)-1):
        s = s + interval_MSE(x[i], t[i-1], t[i])
    return s

def centroid(t1,t2):
    if integrate.quad(f, t1, t2)[0] == 0 or t1 == t2:
        return 0
    else:
        return integrate.quad(lambda t:t*f(t), t1, t2)[0] / integrate.quad(f, t1, t2)[0]

def maxlloyd(t, x):
    for c in range(300):
        # adjust levels
        x[0] = centroid(-float('Inf'), t[0])
        x[-1] = centroid(t[-1], float('Inf'))
        for i in range(1,len(x)-1):
            x[i] = centroid(t[i-1], t[i])

        # adjust thresholds
        for i in range(len(t)):
            t[i] = 0.5 * ( x[i] + x[i+1] )

    e = MSE(t,x)
    print(e)
    return t, x

x_dict = {}
t_dict = {}

for n in range(2,257):  
    print(f"-----s={n}------")  
    m = n + n-1
    interval = np.linspace(-2.8, 2.8, m, endpoint=True)

    x = interval[::2]  # 选择偶数下标项
    t = interval[1::2]  # 选择奇数下标项
    t2, x2 = maxlloyd(t, x)

    list_name_x = f"{n}_bit_x"
    x_dict[list_name_x] = x2

    list_name_t = f"{n}_bit_t"
    t_dict[list_name_t] = t2

print(x_dict)
print(t_dict)

import pickle

file_name = "x_dict.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(x_dict, file)
    
file_name = "t_dict.pkl"
with open(file_name, 'wb') as file:
    pickle.dump(t_dict, file)

 
 
