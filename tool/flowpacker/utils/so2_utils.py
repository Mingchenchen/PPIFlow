import torch
import math

def log(x,y):
    # log_x(y)
    return torch.atan2(torch.sin(y-x), torch.cos(y-x))

def exp(x,y, alt_mask=None):
    # exp_x(y) in [0, 2*pi]
    expmap = (x+y) % (2*math.pi)
    if alt_mask is not None:
        expmap[alt_mask==1] = expmap[alt_mask==1] % math.pi

    return expmap