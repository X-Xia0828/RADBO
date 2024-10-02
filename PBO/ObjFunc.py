import torch
import numpy as np
import zoopt
from zoopt.utils.zoo_global import nan


# domain [-10, 10]
def Dixon_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x = x - 2
    D = len(random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    x1 = eff_dim[:, 0]
    xi = eff_dim[:, 1:]
    xi_minus_1 = eff_dim[:, :-1]
    f = (x1 - 1) ** 2 + ((torch.arange(2, D + 1) * (2 * xi ** 2 - xi_minus_1) ** 2).sum(dim=1))
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - (eff_dim ** 2).sum(dim=1)) / K
    return -(f + dis)


# domain [-50, 50]
def Schwefel_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x = x - 10
    D = len(random_list)
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    f = 418.9829 * D - (eff_dim * torch.sin(torch.sqrt(torch.abs(eff_dim)))).sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - (eff_dim ** 2).sum(dim=1)) / K
    return -(f+dis)



# domain [-10, 10]
def levy_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x_eff = torch.index_select(x, 1, torch.tensor(random_list))
    w = 1 + (x - 1) / 4
    w = torch.index_select(w, 1, torch.tensor(random_list))
    a = torch.sin(torch.pi * w[:, 0]) ** 2
    c = (w[:, -1] - 1) ** 2 * (1 + torch.sin(2 * torch.pi * w[:, -1]) ** 2)
    b = (w[:, :-1] - 1) ** 2 * (1 + 10*torch.sin(torch.pi * w[:, :-1] + 1) ** 2)
    b = b.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - (x_eff ** 2).sum(dim=1)) / K
    return  -(a + b + c + dis)


# domain [-50, 50]
def griewank_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    x = x - 10
    d_e = len(random_list)
    K = 10000
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2 / 4000
    a = a.sum(dim=1)
    b = torch.ones(x.shape[0])
    for i in range(d_e):
        b *= torch.cos(eff_dim[:, i] / torch.sqrt(torch.tensor(i + 1)))
    dis = x[:, :] ** 2
    dis1 = eff_dim[:, :] ** 2    
    dis = (dis.sum(dim=1) - dis1.sum(dim=1)) / K
    return -(a - b + 1 + dis)


# domain [-5.12, 5.12]
def sphere_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x = x - 1
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return -(a + dis)


# domain [-5, 10]
def zakharov_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    K = 10000
    x = x - 1

    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    d_e = eff_dim.size(1)
    a = eff_dim ** 2
    a = a.sum(dim=1)
    par = torch.zeros(x.shape[0])
    for i in range(d_e):
        par += 0.5 * (i + 1) * eff_dim[:, i]
    b = par ** 2
    c = par ** 4
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K

    return -(a + b + c + dis)

# domain [-5, -10]
def rosenbrock_rand(x, random_list):
    if type(x) == zoopt.solution.Solution:
        return nan
    d_e = 10
    K = 10000
    eff_dim = torch.index_select(x, 1, torch.tensor(random_list))
    a = eff_dim ** 2
    a = a.sum(dim=1)
    b = eff_dim[:, 1:] - eff_dim[:, :-1] ** 2
    b = 100 * (b ** 2)
    b = b + (eff_dim[:, 1:] - 1) ** 2
    b = b.sum(dim=1)
    dis = x[:, :] ** 2
    dis = (dis.sum(dim=1) - a) / K
    return -(b + dis)


