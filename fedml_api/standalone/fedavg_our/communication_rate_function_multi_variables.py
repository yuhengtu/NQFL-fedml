import numpy as np

# t 为迭代的次数 (标量)
# sz_k 为client 的个数 (标量)
# n 参数个数（标量）
# Var_S 为上一轮global 的梯度的方差 (vector 参数长度) **
# Var_G 为local client 的梯度的方差 (vector 长度为client个数,参数长度) **
# D 为本次迭代设置的失真约束值 (标量) 
# S_X 本地对全局的估计 (vector 长度为client个数，参数长度)
# M 本地对全局的估计的误差 (vector 长度为client个数，参数长度)
# S 上一轮的全局梯度 (vector 参数长度) **
# P 每个client的压缩误差 (vector 长度为client个数，参数长度)
# Var_W 为高斯随机漫步的方差 (标量)
def RDfucntion(t, sz_k, n, Var_S, Var_G, D, S_X, M, S, P, Var_W):
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

    for k in range(0,sz_k):
        Var_N[k] = np.abs(Var_G[k] - Var_S)
        N[k] = np.abs( np.random.normal(0, Var_N[k]))

    # 算法1：计算local client对global的估计
    #空间分配
    S_X_minus = np.zeros((sz_k, n))  # priori estimate of S by X
    M_minus = np.zeros((sz_k, n))  # priori error estimate of S by X
    mu_SX = np.zeros((sz_k, n))  # innovation of S by X
    K_SX = np.zeros((sz_k, n))  # gain or blending factor of S by X
    #####
    for k in range(0, sz_k):
        S_X_minus[k] = A * S_X[k]  # S_X-
        M_minus[k] = A * M[k] * A + Var_W  # M-
        X[k] = B[k] * S + N[k]  # obversation
        mu_SX[k] = X[k] - B[k] * S_X_minus[k]  # innovation
        #    Var_mu_SX = B * B * M_minus[t] + Var_N
        K_SX[k] = B[k] * M_minus[k] / (B[k] * B[k] * M_minus[k] + Var_N[k])  # gain K
        S_X_new[k] = S_X_minus[k] + K_SX[k] * mu_SX[k]
        M_new[k] = (1 - B[k] * K_SX[k]) * M_minus[k]
    #算法2： Find the optimal level of water-filling
    K_nu = np.full(n, sz_k)
    M_nu = M_new
    nu = 1 / K_nu * (np.sum(1 / M_nu, 0) - (sz_k - 1) / Var_S - 1 / D)
    # k_max = np.where(M_nu == np.max(M_nu, 0))
    k_maxs = []
    M_nu_sub = [M_nu[:, i] for i in range(n)]
    for i in range(n):
        k_maxs.append(np.where(M_nu_sub[i] == np.max(M_nu_sub[i])))

    for i in range(n):
        while 1 / np.max(M_nu_sub[i]) <= nu[i]:
            M_nu_sub[i] = np.delete(M_nu_sub[i], k_maxs[i])
            if len(M_nu_sub[i]) == 0:
                break
            K_nu[i] = K_nu[i] - 1
            nu[i] = 1 / K_nu[i] * (np.sum(1 / M_nu_sub[i]) - (sz_k - 1) / Var_S[i] - 1 / D)
            k_maxs[i] = np.where(M_nu_sub[i] == np.max(M_nu_sub[i]))
    nu_real = 1 / nu + D   #注水线

    P_mines = np.zeros((sz_k, n))
    P_mines[:] = A * A * P[:] + Var_W
    for k in range(0, sz_k):
         P_new[k] = np.max([1 / (1 / M_new[k]) - 1 / (nu_real - D), (1 / Var_N[k] + 1 / P_mines[k])], 0)
    R = np.abs( 0.5 * np.log2(np.abs((np.sum(1 / P_new, 0) - (sz_k - 1) / Var_S) / (np.sum(1 / P_mines, 0) - (sz_k - 1) / Var_S))) + np.sum(0.5 * np.log2(np.abs( (1 - M_new[:] / P_mines[:]) / (1 - M_new[:] / P_new[:]))), 0) )

    # 
    return R, P_new , nu_real, S_X_new, M_new

if __name__ == '__main__':
    sz_t = 100  # 总的迭代层数
    sz_k = 30   # 总的clients个数
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

    # 初始化
    G_global[0] = np.abs(np.random.normal(0, Var_S[0]))
    G_client[0] = np.ones((sz_k, n))
    S_X[0] = 1.0
    M[0, :] = 0.4
    P[0, :] = 0.9
    Var_S[0] = 0.2
    Var_G[0, :] = 0.5
    Var_W    = 0.1

    ### Var_G[t-1, :] 这个参数改为local clients 模型更新后的梯度的方差
    for t in range(1, sz_t):
        R[t], P[t, :], nu_real[t],  S_X[t, :], M[t, :] = RDfucntion(t, sz_k, n, Var_S[t-1], Var_G[t-1, :], D[t-1], S_X[t-1, :], M[t-1, :], G_global[t-1], P[t-1, :], Var_W)
        Var_S[t] = Var_S[t-1] + Var_W
        G_global[t] = np.abs(np.random.normal(0, Var_S[t]))

    #其中R[t]就是第t次迭代所需的传输比特数

    ###最后得到的global的模型的方差为 Var_S[t-1] - D[t] 然后进行随机数生成  np.abs(np.random.normal(0,  Var_S[t-1] - D[t])). 带入模型中计算准确率。
    ###Plan 1： 用np.abs(np.random.normal(0,  Var_S[t-1] - D[t]))随机生成很多个global的模型的梯度，然后对模型的accuracy值进行一个平均，看一下是不是比第一条线优异
 
