import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def quantize(g_vec, input_compress_settings={}):
    compress_settings = {'n': 6} # 默认的压缩参数 6 bit
    compress_settings.update(input_compress_settings)
    n =  compress_settings['n'] 
    
    g_vec = g_vec.float().cpu().numpy()
    mean = np.mean(g_vec)
    std = np.std(g_vec)

    def gaussian(t):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(t - mean)**2 / (2 * std**2))

    # t表示区间划分点，thresholds
    # x表示区间质心，centroid/level，t的长度比x少1（两端点为正负无穷）
    def interval_MSE(x,t1,t2):
        return integrate.quad(lambda t: ((t - x)**2) * gaussian(t), t1, t2)[0]
    # quad 返回一个元组 (result, error)，其中 result 是数值积分的结果，而 error 是估计的误差

    def MSE(t,x):
        s = interval_MSE(x[0], -float('Inf'), t[0]) + interval_MSE(x[-1], t[-1], float('Inf'))
        for i in range(1,len(x)-1):
            s = s + interval_MSE(x[i], t[i-1], t[i])
        return s

    def centroid(t1,t2):
        if integrate.quad(gaussian, t1, t2)[0] == 0 or t1 == t2:
            return 0
        else:
            return integrate.quad(lambda t : t * gaussian(t), t1, t2)[0] / integrate.quad(gaussian, t1, t2)[0]

    def maxlloyd(t, x):
        for c in range(100):
            # adjust levels
            x[0] = centroid(-float('Inf'), t[0])
            x[-1] = centroid(t[-1], float('Inf'))
            for i in range(1,len(x)-1):
                x[i] = centroid(t[i-1], t[i])
            # adjust thresholds
            for i in range(len(t)):
                t[i] = 0.5 * ( x[i] + x[i+1] )
        return t, x

    m = 2 ** n + 2 ** (n) - 1 
    interval = np.linspace(np.min(g_vec), np.max(g_vec), m)
    x = interval[::2]  # 选择偶数下标项
    t = interval[1::2]  # 选择奇数下标项
    t2, x2 = maxlloyd(t, x)

    t2 = torch.tensor(t2).cuda()
    x2 = torch.tensor(x2).cuda()
    g_vec = torch.tensor(g_vec).cuda()
    q_g_vec = torch.full_like(g_vec, x[-1])
    for i in range(len(t)-1, -1, -1):
        # torch.gt, a > b 返回true
        mask = torch.gt(t2[i], g_vec)
        q_g_vec = torch.where(mask, x2[i], q_g_vec)

    return q_g_vec



if __name__ == "__main__":

    # input_data = torch.tensor([-5,-4,-3,-2,-1,0,1,2,3,4,5])

    # 手动指定一个二维向量 [2, 3]
    input_data = torch.tensor([[1.0, 2.0, 3.0],
                    [-4.0, 5.0, -6.0]])

    # 手动指定一个二维向量 [2, 3, 4, 5]
    # input_data = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0],
    #                     [6.0, 7.0, 8.0, 9.0, 10.0],
    #                     [11.0, 12.0, 13.0, 14.0, 15.0],
    #                     [16.0, 17.0, 18.0, 19.0, 20.0]],
                    
    #                    [[-1.0, -2.0, -3.0, -4.0, -5.0],
    #                     [-6.0, -7.0, -8.0, -9.0, -10.0],
    #                     [-11.0, -12.0, -13.0, -14.0, -15.0],
    #                     [-16.0, -17.0, -18.0, -19.0, -20.0]],
                    
    #                    [[0.1, 0.2, 0.3, 0.4, 0.5],
    #                     [0.6, 0.7, 0.8, 0.9, 1.0],
    #                     [1.1, 1.2, 1.3, 1.4, 1.5],
    #                     [1.6, 1.7, 1.8, 1.9, 2.0]]],

    #                   [[[4.0, 3.0, 2.0, 1.0, 0.0],
    #                     [-5.0, -4.0, -3.0, -2.0, -1.0],
    #                     [0.5, 0.4, 0.3, 0.2, 0.1],
    #                     [-1.5, -1.4, -1.3, -1.2, -1.1]],
                    
    #                    [[2.0, 3.0, 4.0, 5.0, 6.0],
    #                     [-7.0, -6.0, -5.0, -4.0, -3.0],
    #                     [0.6, 0.7, 0.8, 0.9, 1.0],
    #                     [-1.0, -0.9, -0.8, -0.7, -0.6]],
                    
    #                    [[-0.5, -0.4, -0.3, -0.2, -0.1],
    #                     [0.5, 0.4, 0.3, 0.2, 0.1],
    #                     [1.0, 0.9, 0.8, 0.7, 0.6],
    #                     [-0.6, -0.7, -0.8, -0.9, -1.0]]]])


    quantized_data = quantize(input_data, {'n': 2})

    print("Original Data:")
    print(input_data)

    print("\nQuantized Data:")
    print(quantized_data)