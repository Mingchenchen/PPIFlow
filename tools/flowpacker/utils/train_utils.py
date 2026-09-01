import torch

def recursive_to(obj, device):
    if isinstance(obj, torch.Tensor):
        if device == 'cpu':
            return obj.cpu()
        return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to(o, device=device) for o in obj]
    elif isinstance(obj, tuple):
        return tuple(recursive_to(o, device=device) for o in obj)
    elif isinstance(obj, dict):
        return {k: recursive_to(v, device=device) for k, v in obj.items()}
    else:
        return obj

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
