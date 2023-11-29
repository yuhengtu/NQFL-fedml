import torch
import os


def quantize(x, input_compress_settings={}):
    compress_settings = {'n': 6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n = compress_settings['n']   # 这里的比特数就不要改了，因为我们用的就是码字的总数带进来，而不是比特数。
    #print('n:{}'.format(n))
    x = x.float()
    x_norm = torch.norm(x, p=2)
    
    sgn_x = ((x > 0).float() - 0.5) * 2
    
    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    xi = (floor_p + margin) / n
    
    Tilde_x = x_norm * sgn_x * xi
    
    return Tilde_x

def quantize_2(x, x_norm, input_compress_settings={}):
    compress_settings = {'n': 6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n = compress_settings['n']   # 这里的比特数就不要改了，因为我们用的就是码字的总数带进来，而不是比特数。
    #print('n:{}'.format(n))
    x = x.float()
    
    sgn_x = ((x > 0).float() - 0.5) * 2
    
    p = torch.div(torch.abs(x), x_norm)
    renormalize_p = torch.mul(p, n)
    floor_p = torch.floor(renormalize_p)
    compare = torch.rand_like(floor_p)
    final_p = renormalize_p - floor_p
    margin = (compare < final_p).float()
    xi = (floor_p + margin) / n
    
    Tilde_x = x_norm * sgn_x * xi
    
    return Tilde_x
