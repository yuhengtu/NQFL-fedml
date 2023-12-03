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

# t的长度比x少1
def estimate(t, x, value):
    for i in range(len(t)):
        if t[i] > value:
            return x[i]
    return x[-1]

def quantize(g_vec, input_compress_settings={}):
    compress_settings = {'n': 6} # 默认的压缩参数 6 bit
    compress_settings.update(input_compress_settings)
    n =  compress_settings['n']
    
    list_name_t = f"{n}_bit_t"
    t = t_dict[list_name_t]

    list_name_x = f"{n}_bit_x"
    x = x_dict[list_name_x]

    g_vec = g_vec.float()
    mean = torch.mean(g_vec)
    std = torch.std(g_vec)
    normalized_g_vec = (g_vec - mean) / std
    
    q_norm_g_vec = [estimate(t, x, value) for value in normalized_g_vec]

    q_g_vec = q_norm_g_vec * std + mean
    
    return q_g_vec
    