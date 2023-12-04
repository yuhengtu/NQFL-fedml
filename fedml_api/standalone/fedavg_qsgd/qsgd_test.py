import torch

def quantize(x, input_compress_settings={}):
    compress_settings = {'n': 6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n =  compress_settings['n']
    #print('n:{}'.format(n))
    x = x.float()
    x_norm = torch.norm(x, p=2) # 计算 L2 范数
    
    sgn_x = ((x > 0).float() - 0.5) * 2 # 得到 x 中每个元素的符号，映射到 -1 或 1。
    
    p = torch.div(torch.abs(x), x_norm) # 元素绝对值 / L2 范数
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p) # 向下取整, 即公式中l
    compare = torch.rand_like(floor_p) # 生成与 floor_p 同样形状的随机张量
    final_p = renormalize_p - floor_p 
    margin = (compare < final_p).float() # 将比较结果映射到 0 或 1
    xi = (floor_p + margin) / n
    
    Tilde_x = x_norm * sgn_x * xi
    
    return Tilde_x

# 手动指定一个二维向量 [2, 3]
# x = torch.tensor([[1.0, 2.0, 3.0],
#                   [-4.0, 5.0, -6.0]])

# 手动指定一个二维向量 [2, 3, 4, 5]
x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0],
                    [6.0, 7.0, 8.0, 9.0, 10.0],
                    [11.0, 12.0, 13.0, 14.0, 15.0],
                    [16.0, 17.0, 18.0, 19.0, 20.0]],
                   
                   [[-1.0, -2.0, -3.0, -4.0, -5.0],
                    [-6.0, -7.0, -8.0, -9.0, -10.0],
                    [-11.0, -12.0, -13.0, -14.0, -15.0],
                    [-16.0, -17.0, -18.0, -19.0, -20.0]],
                   
                   [[0.1, 0.2, 0.3, 0.4, 0.5],
                    [0.6, 0.7, 0.8, 0.9, 1.0],
                    [1.1, 1.2, 1.3, 1.4, 1.5],
                    [1.6, 1.7, 1.8, 1.9, 2.0]]],

                  [[[4.0, 3.0, 2.0, 1.0, 0.0],
                    [-5.0, -4.0, -3.0, -2.0, -1.0],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [-1.5, -1.4, -1.3, -1.2, -1.1]],
                   
                   [[2.0, 3.0, 4.0, 5.0, 6.0],
                    [-7.0, -6.0, -5.0, -4.0, -3.0],
                    [0.6, 0.7, 0.8, 0.9, 1.0],
                    [-1.0, -0.9, -0.8, -0.7, -0.6]],
                   
                   [[-0.5, -0.4, -0.3, -0.2, -0.1],
                    [0.5, 0.4, 0.3, 0.2, 0.1],
                    [1.0, 0.9, 0.8, 0.7, 0.6],
                    [-0.6, -0.7, -0.8, -0.9, -1.0]]]])

print("Original Vector:")
print(x)

quantized_x = quantize(x, {'n': 6})

print("\nQuantized Vector:")
print(quantized_x)
