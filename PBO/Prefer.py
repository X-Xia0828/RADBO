import torch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.sampling import IIDNormalSampler
import numpy as np
from cmaes import CMA
import warnings

warnings.filterwarnings("ignore")

def updata_dataset(dataset, x, xx, preInfo):
    
    dataset['x'] = torch.cat([dataset['x'], torch.cat([x, xx], -1)], 0)
    dataset['x'] = torch.cat([dataset['x'], torch.cat([xx, x], -1)], 0)
    # print(torch.tensor([preInfo[0]]).reshape(1,-1))
    dataset['y'] = torch.cat([dataset['y'], torch.tensor([preInfo[0]]).reshape(1,-1)], 0)
    dataset['y'] = torch.cat([dataset['y'], torch.tensor([preInfo[1]]).reshape(1,-1)], 0)
    return dataset

def max_variance_next_xx(next_x, model, dim, bounds, optimization_type = 'cmaes'):

    def variance(xx):
        xx = torch.from_numpy(xx)
        points = torch.cat((next_x.squeeze(0), xx), -1).unsqueeze(0)
        points = torch.as_tensor(points, dtype=torch.float32)
        y = torch.tensor(model(points).stddev)
        return y.numpy()

    optimization_type = 'cmaes'
    if optimization_type == 'cmaes':
        optimizer = CMA(mean=np.zeros(dim), sigma=1.3, bounds=bounds.numpy(), population_size=2)
        score = 1e10
        next_xx = 'NULL'
        for generation in range(20):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = -variance(x)
                solutions.append((x, value))
                if value <= score:
                    score = value
                    next_xx = x
            optimizer.tell(solutions)
    else:
        score = 'NULL'
        next_xx = 'NULL'
        print('Please input correct optimization type, such as \'direct\' or \'cmaes\'')

    next_xx = torch.from_numpy(next_xx).unsqueeze(0)

    return next_xx




def soft_copeland_score_next_x(model, seed , objDim, bounds, points_number, optimization_type='cmaes'):
    def scs_objective_function(x):
        # this enables multiple points to be sampled at the same time
        # build the grid
        x = torch.from_numpy(x)
        grid = torch.rand(points_number, objDim) * (bounds.t()[1] - bounds.t()[0]) + bounds.t()[0]

        # build points set which need to be sampled
        x_ex = x*torch.ones([grid.shape[0], grid.shape[1]])
        sample_points = torch.cat([x_ex, grid], -1)
        # sample_points = torch.as_tensor(sample_points, dtype=torch.float32)

        # sample and compute
        sampler = IIDNormalSampler(1, seed=seed)
        posterior = model.posterior(sample_points)
        values = logistic(torch.tensor(sampler(posterior)))
        integral_value = torch.mean(values)

        return integral_value.numpy()
    if optimization_type == 'cmaes':
        optimizer = CMA(mean=np.zeros(objDim), sigma=1.3, bounds=bounds.numpy(), population_size=2)
        score = 1e10
        optimal_point = 'NULL'
        for generation in range(20):
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                value = -scs_objective_function(x)
                solutions.append((x, value))
                if value <= score:
                    score = value
                    optimal_point = x
            optimizer.tell(solutions)
    else:
        score = 'NULL'
        optimal_point = 'NULL'
        print('Please input correct optimization type, such as \'direct\' or \'cmaes\'')
    
    optimal_point = torch.from_numpy(optimal_point).unsqueeze(0)
    
    return optimal_point

def fit_model_gp(dataset):
    dataset_x = dataset['x']
    dataset_y = dataset['y']
    model = SingleTaskGP(dataset_x, dataset_y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)
    return model

def logistic(x):
    return 1 / (1 + torch.exp((-1) * x))


def get_preference_infor(x1, x2, objFunc):
    fx1 = torch.tensor(objFunc.evaluate_true(x1))
    fx2 = torch.tensor(objFunc.evaluate_true(x2))
    regret1 = fx1-objFunc.optimal
    regret2 = fx2-objFunc.optimal
    regret = torch.max(torch.cat((regret1, regret2), 0), 0).values
    
    preference = logistic(fx1-fx2)-torch.rand(fx1.shape[0]).reshape(fx1.shape[0], 1)
    
    for i, cc in enumerate(preference):
        if cc >= 0:
            preference[i] = 1.
        else:
            preference[i] = 0.
    return preference, regret


def init_dataset(initNum, objDim, bounds, objFunc):
    a = torch.rand(initNum, objDim)
    a = a*(bounds[1]-bounds[0])+bounds[0]
    aa = torch.rand(initNum, objDim)
    aa = aa*(bounds[1]-bounds[0])+bounds[0]
    preference, regret = get_preference_infor(torch.cat([a, aa], 0), torch.cat([aa, a], 0), objFunc)
   
    x = torch.cat([a, aa], 1)
    xx = torch.cat([aa, a], 1)

    x = torch.cat([x,xx], 0)
    
    dataset = {'x': x, 'y': preference}
    return dataset, regret