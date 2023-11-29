##################
###
### 
###
#本函数为运行quantize_rate(round, num_clients, G_global, G_client, Var_S, Var_G, S_X , M , P)这个函数
#输入为 #round 为本次迭代轮次数。这里我觉得不用了，因为每一轮跑一次这个函数
        #num_clients 为总的local client数。 是一个标量
        #G_global 为全局的梯度数值。这个数值应该可以直接输入，就是上一轮global下行来的数据。是n维，n是local client的梯度的维数
        #G_client 为local client的梯度数值，为k*n维，k为client的数量，n为每个client梯度的维度d。这个数据应该也是可以直接输入的
        #Var_S 为全局的梯度方差。这个是一个统计量 是n维
        #Var_G 为本地的梯度方差。这个是一个统计量 在fedavg_api里有将Var_S初始化为第一轮训练得到的全局梯度方差的方法。以后的可以通过统计或者什么办法得到这个方差吧。是k*n维
        ##还需要上一个round的跑出来的一些数值：
        #S_X       = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计 如果跟着时间更新不保存是k*n维，如果把每次迭代的值都记下则为t*k*n维
        #M         = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计的误差 如果跟着时间更新不保存是k*n维，如果把每次迭代的值都记下则为t*k*n维
        #P         = np.zeros((sz_t,sz_k, n))    # 每个client的压缩误差 如果跟着时间更新不保存是k*n维，如果把每次迭代的值都记下则为t*k*n维
        ##在整个过程开始前，对其进行空间分配，初始化，后续的值有本函数得到。
        #S_X[0,:,:]    =  0.4
        #    for k in range(0, sz_k):
        #M[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.1))
        #P[0, k] = M[0, k] + np.abs(np.random.normal(0, 0.2))
        # 上文是初始化 M 和 P

#输出为 return R, R_client, SSSS, S_X, M , P
    #R 为和码率，即和比特数. 是n维，即为在第n维上，k个local client的码率和
    #R_client 为每个local client 分配到的比特数 是K*n维，在k个local client 的每个维度输出一个量化比特数
    #SSSS 为每个local client 使用的码字数 是K*n维，在k个local client 的每个维度输出一个量化码字数
    #S_X ， M， P 为中间变量，在下一次使用本函数时需要使用，每次运行本函数，更新一次这三个变量，第一次由初始化得到。 均如果跟着时间更新不保存是k*n维，如果把每次迭代的值都记下则为t*k*n维
#####################################
#PS：如果不能对每个维度进行不同的量化比特，那我们就使用n个维度的量化比特平均值。
##########################
##所以本函数最后的使用方法是：使用输出的SSSS值，带入client.py的qsgd_quantize 函数和 QSGD.py 的quantize函数中。
##########################


import numpy as np
from numpy import sort
from numpy import argsort


def RDfucntion(sz_k, n, D, Var_S, Var_G, G , S_X, M,  P, Var_W):  ## 0416改：这里加了一个G，为local的梯度值，是一个k维数组
    A = 1
    B = np.ones((sz_k))

    # 计算local client 的梯度方差和global的梯度方差的差值 成为Var_N
    # 其随机生成的高斯随机变量的值为N
    Var_N = np.zeros(sz_k)
    S_X_new = np.zeros(sz_k)
    M_new = np.zeros(sz_k)
    P_new = np.zeros(sz_k)
    N = np.zeros(sz_k)
    X = np.zeros(sz_k)
    Temp_P = np.zeros(sz_k)  # 0416改：这里加了一个变量
    aa = np.zeros(sz_k)  # 0416改：这里加了一个变量
    R_client = np.zeros(sz_k)
    R_client_temp = np.zeros(sz_k)

    for k in range(0, sz_k):
        Var_N[k] = np.abs(Var_G[k] - Var_S)  ##make sure that var_G > Var_S
        N[k] = np.random.normal(0, Var_N[k])  ## 0416改：这里删了一个abs

    # 算法1：计算local client对global的估计
    # 空间分配
    S_X_minus = np.zeros(sz_k)  # priori estimate of S by X
    M_minus = np.zeros(sz_k)  # priori error estimate of S by X
    mu_SX = np.zeros(sz_k)  # innovation of S by X
    K_SX = np.zeros(sz_k)  # gain or blending factor of S by X
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



    P_mines = np.zeros(sz_k)
    P_mines[:] = A * A * P[:] + Var_W

    for k in range(0, sz_k):
        aa[k] = 1 / (1 / M_new[k] - 1 / Var_N[k] - 1 / P_mines[k])
    power = np.sum(1 / Var_N[:] + 1 / P_mines[:], 0) - (1 / D) - (sz_k - 1) / Var_S
    ww = np.ones(sz_k)

    water_level, index, value, si, height = GWF(power, aa, ww)
    nu_real = 1 / water_level + D

    for k in range(0, sz_k):
        if 1 / M_new[k] - water_level > 0:
            Temp_P[k] = max(water_level - 1 / M_new[k] + 1 / Var_N[k] + 1 / P_mines[k], 0)
            P_new[k] = 1 / (1 / Var_N[k] + 1 / P_mines[k] - Temp_P[k])
            R_client_temp[k] = 0.5 * np.log2(Var_S / P_new[k])
        else:
            Temp_P[k] = 0
            P_new[k] = 1 / (1 / Var_N[k] + 1 / P_mines[k])
            R_client_temp[k] = 0

    R = 0.5 * np.log2( np.maximum( 
        (np.sum(1 / P_new[:]) - (sz_k - 1) / Var_S) / (np.sum(1 / P_mines[:]) - (sz_k - 1) / Var_S) , 1)  ) + np.sum(
        0.5 * np.log2( np.maximum( (1 - M_new[:] / P_mines[:]) / (1 - M_new[:] / P_new[:]), 1 )  ) )

    for k in range(0, sz_k):
        R_client[k] = np.maximum((R - np.sum(R_client_temp[:], 0)) / sz_k + R_client_temp[k], 0)


    #print(R_client)
    return R, R_client,  nu_real, S_X_new, M_new, P_new


# Work in Python 3.9.1
# Dependences: Numpy




def GWF(power, gain, weight):
    power = power
    count = 0
    a = sort(gain)[::-1]
    w = weight
    height = sort(1 / (w * a))
    # print(height)
    ind = argsort(1 / (w * a))
    weight = weight[ind]
    # print(weight)

    original_size = len(a) - 1  # size of gain array, i.e., total # of channels.
    channel = len(a) - 1
    isdone = False

    # print('*' * 30, 'enter while')
    # while isdone == False:
    #     Ptest = 0  # Ptest is total 'empty space' under highest channel under water.
    #     for i in range(channel):
    #         Ptest += (height[channel] - height[i]) * weight[i]
    #         # print(Ptest)
    #         # print(height)
    #     if (power - Ptest) >= 0:  # If power is greater than Ptest, index (or k*) is equal to channel.
    #         index = channel  # Otherwise decrement channel and run while loop again.
    #         # print(index)
    #         break
                
    #     channel -= 1



###############
###############
    index = channel
    for j in range (0,len(a) - 1):
        Ptest = 0
        for i in range(channel):
            if (power - Ptest)  < 0 :
                Ptest += (height[channel] - height[i]) * weight[i]
            else:  # If power is greater than Ptest, index (or k*) is equal to channel.
                index = channel 
                break
        if (power - Ptest) >= 0:
            break
        channel -= 1




    # print('*' * 30, 'left while')
    # print('index = ' + str(index))
    # print(height)
    value = power - Ptest  # 'value' is P2(k*)
    # print(value)
    water_level = value / np.sum([weight[range(index + 1)]]) + height[index]
    # print(weight[range(index)])
    # print('sum = ' + str(np.sum(weight[range(index)])))
    si = (water_level - height) * weight
    si[si < 0] = 0
    return water_level, index, value, si, height


#if __name__ == '__main__':


    #####在使用这个函数之前，我们需要得到
    
    #G_global 为全局的梯度数值。这个数值应该可以直接输入，就是上一轮global下行来的数据
    #G_client 为local client的梯度数值，为k*n维，k为client的数量，n为每个client梯度的维度d。这个数据应该也是可以直接输入的
    #Var_S 为全局的梯度方差。这个是一个统计量
    #Var_G 为本地的梯度方差。这个是一个统计量 在fedavg_api里有将Var_S初始化为第一轮训练得到的全局梯度方差的方法。以后的可以通过统计或者什么办法得到这个方差吧。

    ##还需要上一个round的跑出来的一些数值：
    #S_X       = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计
    #M         = np.zeros((sz_t,sz_k, n))    # 本地对全局的估计的误差
    #P         = np.zeros((sz_t,sz_k, n))   # 每个client的压缩误差
    ##在整个过程开始前，对其进行空间分配，初始化，后续的值有本函数得到。
    #S_X[0,:,:]    =  0.4
    #    for k in range(0, sz_k):
    #M[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.1))
    #P[0, k] = M[0, k] + np.abs(np.random.normal(0, 0.2))
    # 上文是初始化 M 和 P
    #

    ####
    #这个函数，现在是求出qsgd中的q值。输出就是每个local client的每个维度的量化码字数Q，也就是之前程序中qsgd的q。
    # return 的 SSSS 就是 Q
def quantize_rate(round, num_clients, n, t, G_global, G_client, Var_S, Var_G, S_X , M , P):
    #round 为本次迭代轮次数。这里我觉得不用了，因为每一轮跑一次这个函数
    #num_clients 为总的local client数

    
    sz_t = round  # 总的迭代层数
    sz_k = num_clients  # 总的clients个数


    D         = np.full(n,0.2)   # 每次迭代允许的失真约束
    R = np.zeros((n))           # 和速率
    nu_real   = np.zeros((sz_t,n))          # 注水线

    #G_value = np.zeros((sz_t, sz_k, n))
    R_client = np.zeros((sz_k, n))  # 每个local client 的 communication bits。
    SSSS = np.zeros((sz_k, n))    #每个local client 的 量化bit值
    Var_W = 0.1# 随机漫步的方差数
    # 初始化
    #G_global[0] = np.random.normal(0, Var_S[0])
    # G_client[0, k] = np.abs (1 +  np.random.normal(0,1))
    # Var_G[0, :] = 0.4
    # for t in range(0, 1):
    #for k in range(0, sz_k):
    #    Var_G[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.5))
     #   G_value[0, k] = np.random.normal(0, Var_G[0, k])

     ###上述的初始化过程，应该由数据集得到

    ### Var_G[t-1, :] 这个参数改为local clients 模型更新后的梯度的方差
    for nn in range(0, n):  ##对每个维度进行求解需要量化的bit数,如果是在d维上整个向量输入，那就分成d个数分别计算？
        # 下一行多加了个参数 G_value
        R[nn], R_client[:, nn], nu_real[t, nn],  S_X[:, nn], M[:, nn], P[:, nn]   = RDfucntion(sz_k, nn, D[nn], 
                                                                                    Var_S[nn], Var_G[:, nn],  G_client[:, nn],
                                                                                S_X[:, nn], M[: , nn], P[:, nn], Var_W)
        Var_S[nn] = Var_S[nn] + Var_W
        G_global[nn] = (np.random.normal(0, Var_S[nn]))
        for k in range(0, sz_k):
            Var_G[k,nn] = Var_S[nn] + np.abs(np.random.normal(0, 0.5))
            G_client[k,nn] = np.random.normal(0, Var_G[k,nn])
            if t >= 2 :
                SSSS[k][nn] = (2 ** R_client[k][nn]).astype(int) + 1   #QSGD中的码字数，用于量化后的通信开销
            else:
                SSSS[k][nn] = 1
    #print(SSSS)
    return R, R_client, SSSS, S_X, M , P

    ####
    #R 为和速率，即和比特数
    #R_client 为每个local client 分配到的比特数
    #SSSS 为每个local client 使用的码字数
    #S_X ， M， P 为中间变量，在下一次使用本函数时需要使用，每次运行本函数，更新一次这三个变量，第一次由初始化得到。




# t 为迭代的次数
# sz_k 为client 的个数
# Var_S 为上一轮global 的梯度的 方差
# Var_G 为local client 的梯度的方差
# D 为本次迭代设置的失真约束值
# Var_W 为高斯随机漫步的方差
# G_value 为第k个local client上的梯度数值
# R_client 为第k个local client上使用的量化bit数 带入
# SSSS为码字数，即Adaptive SGD中公式（6）中的S

