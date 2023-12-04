import torch
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

import pickle
file_name = "t_dict.pkl"
with open(file_name, 'rb') as file:
    t_dict = pickle.load(file)

file_name = "x_dict.pkl"
with open(file_name, 'rb') as file:
    x_dict = pickle.load(file)

def quantize(g_vec, input_compress_settings={}):
    compress_settings = {'n': 6} # 默认的压缩参数 6 bit
    compress_settings.update(input_compress_settings)
    n =  compress_settings['n'] 
    
    list_name_t = f"{n}_bit_t"
    t = torch.tensor(t_dict[list_name_t]).cuda()

    list_name_x = f"{n}_bit_x"
    x = torch.tensor(x_dict[list_name_x]).cuda()

    g_vec = g_vec.float().cuda()
    mean = torch.mean(g_vec)
    std = torch.std(g_vec)
    g_vec = (g_vec - mean) / std

    # t的长度比x少1
    # def estimate(t, x, value):
    #     for i in range(len(t)):
    #         if t[i] > value:
    #             return x[i]
    #     return x[-1]
    q_g_vec = torch.full_like(g_vec, x[-1])
    for i in range(len(t)-1, -1, -1):
        # torch.gt, a > b 返回true
        mask = torch.gt(t[i], g_vec)
        q_g_vec = torch.where(mask, x[i], q_g_vec)
    
    q_g_vec = q_g_vec * std + mean
    
    return q_g_vec





# t的长度比x少1
# def estimate(t, x, value):
#     for i in range(len(t)):
#         if t[i] > value:
#             return x[i]
#     return x[-1]

# def quantize(g_vec, input_compress_settings={}):
#     compress_settings = {'n': 6} # 默认的压缩参数 6 bit
#     compress_settings.update(input_compress_settings)
#     n =  compress_settings['n']
    
#     list_name_t = f"{n}_bit_t"
#     t = t_dict[list_name_t]

#     list_name_x = f"{n}_bit_x"
#     x = x_dict[list_name_x]

#     g_vec = g_vec.float()
#     mean = torch.mean(g_vec)
#     std = torch.std(g_vec)
#     normalized_g_vec = (g_vec - mean) / std
    
#     q_norm_g_vec = [estimate(t, x, value) for value in normalized_g_vec]

#     q_g_vec = q_norm_g_vec * std + mean
    
#     return q_g_vec
    





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