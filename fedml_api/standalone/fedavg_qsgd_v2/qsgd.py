import torch

# 变7
def quantize(x, input_compress_settings={}):
    compress_settings = {'n': 6} # 默认的压缩参数 6 bit
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    # 相比v1变
    # n = torch.ceil(compress_settings['n']) + 1  # 向上取整并加一
    # 这里改了一下比特数，为2^n个码字数量。
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
    