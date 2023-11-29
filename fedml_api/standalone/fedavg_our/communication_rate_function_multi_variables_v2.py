# t 为迭代的次数
# sz_k 为client 的个数
# Var_S 为上一轮global 的梯度的 方差
# Var_G 为local client 的梯度的方差
# D 为本次迭代设置的失真约束值
# Var_W 为高斯随机漫步的方差

import numpy as np
from numpy import sort
from numpy import argsort


def RDfucntion(t, sz_k, n, Var_S, Var_G, D, S_X, M, S, P, Var_W, G):  ## 0416改：这里加了一个G，为local的梯度值，是一个k维数组
    A = 1
    B = np.ones((sz_k, n))

    # 计算local client 的梯度方差和global的梯度方差的差值 成为Var_N
    # 其随机生成的高斯随机变量的值为N
    Var_N = np.zeros((sz_k, n))
    S_X_new = np.zeros((sz_k, n))
    M_new = np.zeros((sz_k, n))
    P_new = np.zeros((sz_k, n))
    N = np.zeros((sz_k, n))
    X = np.zeros((sz_k, n))
    # A_alg2 = np.zeros(sz_k)    # 0416改：这里加了一个变量
    # W_alg2 = np.zeros(sz_k)    # 0416改：这里加了一个变量
    # Var_W_alg2 = np.zeros(sz_k) # 0416改：这里加了一个变量
    Temp_P = np.zeros((sz_k, n))  # 0416改：这里加了一个变量
    aa = np.zeros((sz_k, n))  # 0416改：这里加了一个变量
    R_client = np.zeros((sz_k, n))
    R_client_temp = np.zeros((sz_k, n))

    for k in range(0, sz_k):
        Var_N[k] = np.abs(Var_G[k] - Var_S)  ##make sure that var_G > Var_S
        N[k] = np.random.normal(0, Var_N[k])  ## 0416改：这里删了一个abs

    # 算法1：计算local client对global的估计
    # 空间分配
    S_X_minus = np.zeros((sz_k, n))  # priori estimate of S by X
    M_minus = np.zeros((sz_k, n))  # priori error estimate of S by X
    mu_SX = np.zeros((sz_k, n))  # innovation of S by X
    K_SX = np.zeros((sz_k, n))  # gain or blending factor of S by X
    #####
    for k in range(0, sz_k):
        S_X_minus[k] = A * S_X[k]  # S_X-
        M_minus[k] = A * M[k] * A + Var_W  # M-
        # X[k] = B[k] * S + N[k]  # obversation   ### 0416改：这一句不要了，observation就是G
        mu_SX[k] = G[k] - B[k] * S_X_minus[k]  # innovation  ## 0416改：这里的X改为G
        #    Var_mu_SX = B * B * M_minus[t] + Var_N
        K_SX[k] = B[k] * M_minus[k] / (B[k] * B[k] * M_minus[k] + Var_N[k])  # gain K
        S_X_new[k] = S_X_minus[k] + K_SX[k] * mu_SX[k]
        M_new[k] = (1 - B[k] * K_SX[k]) * M_minus[k]

        ## 0416改：下面三行是新增的
        # A_alg2[k] = (A * Var_N[k])/(M_minus[k] + Var_N[k])    # 0416改：新增
        # W_alg2[k] = (M_minus[k] / (M_minus[k]+Var_N[k]　))*　G[k]   # 0416改：新增
        # Var_W_alg2[k] = (M_minus[k] / (M_minus[k]+Var_N[k]　)) *　(M_minus[k] / (M_minus[k]+Var_N[k]　))  * Var_G[k]

    P_mines = np.zeros((sz_k, n))
    P_mines[:] = A * A * P[:] + Var_W

    for k in range(0, sz_k):
        aa[k] = 1 / (1 / M_new[k] - 1 / Var_N[k] - 1 / P_mines[k])
    power = np.sum(1 / Var_N[:] + 1 / P_mines[:], 0) - (1 / D) - (sz_k - 1) / Var_S

    ww = np.ones((sz_k, n))

    water_level, index, value, si, height = GWF(power, aa, ww)
    nu_real = 1 / water_level + D
    # print('water_level = ' + str(water_level))
    # print('index = ' + str(index))
    # print('value = ' + str(value))
    # print('si = ' + str(si))
    # print('height = ' + str(height))
    for i in range(n):
        for k in range(0, sz_k):
            if 1 / M_new[k, i] - water_level[i] > 0:
                Temp_P[k, i] = max(water_level[i] - 1 / M_new[k, i] + 1 / Var_N[k, i] + 1 / P_mines[k, i], 0)
                P_new[k, i] = 1 / (1 / Var_N[k, i] + 1 / P_mines[k, i] - Temp_P[k, i])
                R_client_temp[k, i] = 0.5 * np.log2(Var_S[i] / P_new[k, i])
            else:
                Temp_P[k, i] = 0
                P_new[k, i] = 1 / (1 / Var_N[k, i] + 1 / P_mines[k, i])
                R_client_temp[k, i] = 0
        # P_new[k] = max(1 / (1 / M_new[k]) - 1 / (nu_real - D), (1 / Var_N[k] + 1 / P_mines[k]))
    # R = np.abs( 0.5 * np.log2(np.abs((np.sum(1 / P_new[:]) - (sz_k - 1) / Var_S) / (np.sum(1 / P_mines[:]) - (sz_k - 1) / Var_S))) + np.sum(0.5 * np.log2(np.abs( (1 - M_new[:] / P_mines[:]) / (1 - M_new[:] / P_new[:]))) ) )
    #print(Temp_P)
    R = 0.5 * np.log2(
        (np.sum(1 / P_new[:], 0) - (sz_k - 1) / Var_S) / (np.sum(1 / P_mines[:], 0) - (sz_k - 1) / Var_S)) + np.sum(
        0.5 * np.log2((1 - M_new[:] / P_mines[:]) / (1 - M_new[:] / P_new[:])), 0)
    # 研究一下 每一个client的R是多少
    for k in range(0, sz_k):
        R_client[k] = np.maximum((R - np.sum(R_client_temp[:], 0)) / sz_k + R_client_temp[k], 0)

    #print(R_client)
    return R, P_new, nu_real, S_X_new, M_new, R_client


# Work in Python 3.9.1
# Dependences: Numpy




def GWF(power, gain, weight):
    power = power
    count = 0
    a = sort(gain, 0)[::-1, :]
    w = weight
    height = sort(1 / (w * a), 0)
    # print(height)
    ind = argsort(1 / (w * a), 0)
    # weight = weight[ind]
    for i in range(weight.shape[1]):
        weight[:, 0] = weight[:, 0][ind[:, 0]]
    # print(weight)

    original_size = len(a) - 1  # size of gain array, i.e., total # of channels.
    channel = np.full(a.shape[1], a.shape[0] - 1)
    isdone = False
    index = np.zeros(a.shape[1])
    Ptest = np.zeros(a.shape[1])
    print('*' * 30, 'in for')
    for j in range(a.shape[1]):
        while isdone == False:
            Ptest[j] = 0  # Ptest is total 'empty space' under highest channel under water.
            for i in range(channel[j]):
                Ptest[j] += (height[channel[j], j] - height[i, j]) * weight[i, j]
                # print(Ptest)
                # print(height)
            if (power[j] - Ptest[j]) >= 0:  # If power is greater than Ptest, index (or k*) is equal to channel.
                index[j] = channel[j]  # Otherwise decrement channel and run while loop again.
                # print(index)
                break

            channel[j] -= 1
    print('*' * 30, 'out for')
    # print('index = ' + str(index))
    # print(height)
    value = power - Ptest  # 'value' is P2(k*)
    # print(value)
    water_level = np.zeros(value.shape[0])
    for i in range(value.shape[0]):
        water_level[i] = value[i] / np.sum([weight[range(int(index[i] + 1)), i]]) + height[int(index[i]), i]
    # print(weight[range(index)])
    # print('sum = ' + str(np.sum(weight[range(index)])))
    si = (water_level - height) * weight
    si[si < 0] = 0
    return water_level, index, value, si, height


if __name__ == '__main__':
    sz_t = 200  # 总的迭代层数
    sz_k = 30  # 总的clients个数
    n = 10

    # 空间分配
    G_global  = np.zeros((sz_t, n))    # 全局的梯度值
    G_client  = np.zeros((sz_t,sz_k, n))   # 本地的梯度值
    Var_S     = np.zeros((sz_t, n))              # 全局的梯度方差    ###
    Var_G     = np.zeros((sz_t,sz_k, n))       # 本地的梯度方差
    D         = np.full(sz_t, 0.1)   # 每次迭代允许的失真约束
    S_X       = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计
    M         = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计的误差
    nu_real   = np.zeros((sz_t, n))          # 注水线
    P         = np.zeros((sz_t, sz_k, n))   # 每个client的压缩误差
    R         = np.zeros((sz_t, n))           # 和速率

    G_value = np.zeros((sz_t, sz_k, n))
    R_client = np.zeros((sz_t, sz_k, n))
    SSSS = np.zeros((sz_t, sz_k, n))

    # 初始化
    G_global[0] = np.random.normal(0, Var_S[0])

    S_X[0] = 0.4
    Var_S[0] = 0.3
    for k in range(0, sz_k):
        M[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.1))
        P[0, k] = M[0, k] + np.abs(np.random.normal(0, 0.2))
        # G_client[0, k] = np.abs (1 +  np.random.normal(0,1))

    # Var_G[0, :] = 0.4
    Var_W = 0.02
    # for t in range(0, 1):
    for k in range(0, sz_k):
        Var_G[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.5))
        G_value[0, k] = np.random.normal(0, Var_G[0, k])

    ### Var_G[t-1, :] 这个参数改为local clients 模型更新后的梯度的方差
    for t in range(1, sz_t):
        # 下一行多加了个参数 G_value
        R[t], P[t, :], nu_real[t], S_X[t, :], M[t, :], R_client[t, :] = RDfucntion(t, sz_k, n, Var_S[t - 1],
                                                                                   Var_G[t - 1, :], D[t - 1],
                                                                                   S_X[t - 1, :], M[t - 1, :],
                                                                                   G_global[t - 1], P[t - 1, :], Var_W,
                                                                                   G_value[t, :])
        Var_S[t] = Var_S[t - 1] + Var_W
        G_global[t] = (np.random.normal(0, Var_S[t]))
        for k in range(0, sz_k):
            Var_G[t, k] = Var_S[t] + np.abs(np.random.normal(0, 0.5))
            G_value[t, k] = np.random.normal(0, Var_G[t, k])
            if t > 2 :
                SSSS[t][k] = (2 ** R_client[t][k]).astype(int) + 1   #QSGD中的码字数，用于量化后的通信开销
            else:
                SSSS[t][k] = 32
    print(SSSS)


    # 其中R[t]就是第t次迭代所需的传输比特数



# t 为迭代的次数
# sz_k 为client 的个数
# Var_S 为上一轮global 的梯度的 方差
# Var_G 为local client 的梯度的方差
# D 为本次迭代设置的失真约束值
# Var_W 为高斯随机漫步的方差
# G_value 为第k个local client上的梯度数值
# R_client 为第k个local client上使用的量化bit数 带入
# SSSS为码字数，即Adaptive SGD中公式（6）中的S

