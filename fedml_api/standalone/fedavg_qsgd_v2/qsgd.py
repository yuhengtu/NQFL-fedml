import torch

def quantize(x, input_compress_settings={}):
    compress_settings = {'n': 6}
    compress_settings.update(input_compress_settings)
    #assume that x is a torch tensor
    
    n = torch.ceil(compress_settings['n']) + 1  # 这里改了一下比特数，为2^n个码字数量。
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