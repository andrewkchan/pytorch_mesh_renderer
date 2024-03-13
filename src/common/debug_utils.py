import torch

def debug_tensor(tensor, msg=""):
    torch.set_printoptions(profile="full", linewidth=200)
    print("[debug tensor] {}".format(msg))
    print(tensor)
    torch.set_printoptions(profile="default", linewidth=80)

def check_isnan_isinf(tensor, msg=""):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        raise ValueError(msg)