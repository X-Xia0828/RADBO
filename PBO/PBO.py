# PBO MAX

import torch
import numpy as np
import os
import argparse
import warnings
import random
import Functions as tfs
import Prefer as pbo

torch.set_printoptions(profile="full")
np.set_printoptions(threshold=np.inf)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, nargs='?')
parser.add_argument('--objDim', type=int, default=10, nargs='?')
parser.add_argument('--effDim', type=int, default=10, nargs='?')
parser.add_argument('--budget', type=int, default=100, nargs='?')
parser.add_argument('--epsilon', type=float, default=0, nargs='?')
parser.add_argument('--objectiveFunction', type=str, default='Sphere', nargs='?')
parser.add_argument('--beginNum', type=int, default=5, nargs='?')
args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(0)
torch.set_default_dtype(torch.double)


objDim = args.objDim
effDim = args.effDim
budget = args.budget
beginNum = args.beginNum
optimization_type = 'cmaes'
int_number = 500
epsilon = args.epsilon
objfunctions = args.objectiveFunction
if objfunctions == 'Sphere':
    objectiveFunction = tfs.Sphere(objDim=objDim, effDim=effDim)
elif objfunctions == 'Levy':
    objectiveFunction = tfs.Levy(objDim=objDim, effDim=effDim)
elif objfunctions == 'Rosenbrock':
    objectiveFunction = tfs.Rosenbrock(objDim=objDim, effDim=effDim)
elif objfunctions == 'Griewank':
    objectiveFunction = tfs.Griewank(objDim=objDim, effDim=effDim)
elif objfunctions == 'Schwefel':
    objectiveFunction = tfs.Schwefel(objDim=objDim, effDim=effDim)
elif objfunctions == 'Dixon':
    objectiveFunction = tfs.Dixon(objDim=objDim, effDim=effDim)
bounds = torch.stack([torch.ones(objDim) * (-1), torch.ones(objDim)])
bounds_prefer = torch.tensor([[row[i] for row in bounds] for i in range(len(bounds[0]))])
directory_path = f".\\result\\PBO\\{objfunctions}-{objDim}-{budget}bg{beginNum}\\"
os.makedirs(directory_path, exist_ok=True)  # check if the folder exists
bestSofar = []
rightNum = []
# Initialize the dataset
dataset, regrets= pbo.init_dataset(beginNum, objDim, bounds, objectiveFunction)

for i in range(regrets.shape[0]):
    bestSofar.append(regrets.max().item())

# Fit a GP
gp_model = pbo.fit_model_gp(dataset)
for i in range(budget-beginNum):
    sample_seed = int(torch.rand(1) * 10000)
    # Use soft-Copeland score to find x_next
    x_next = pbo.soft_copeland_score_next_x(gp_model, sample_seed, objDim, bounds_prefer,
                                 int_number, optimization_type)
    # Use max variance to find x'_next
    xx_next = pbo.max_variance_next_xx(x_next, gp_model, objDim, bounds_prefer, optimization_type)
    # Get preference information
    preInfo, r = pbo.get_preference_infor(torch.cat([x_next, xx_next], 0), torch.cat([xx_next, x_next], 0), objectiveFunction)
    regrets= torch.cat((regrets, r), 0)
    for _ in range(r.shape[0]):
        bestSofar.append(max(bestSofar[-1], r.max().item()))
    # Updata the dataset 
    dataset = pbo.updata_dataset(dataset, x_next, xx_next, preInfo)
    
    right = 0
    for datax, datay in zip(dataset['x'], dataset['y']):
        x1 = datax[:objDim]
        x2 = datax[objDim:]
        y = 0.
        # use only for computing accuracy of the preferences.
        if objectiveFunction.evaluate_true(x1) > objectiveFunction.evaluate_true(x2):
            y = 1.
        if y == datay.item():
            right += 1
    rightNum.append(right)
    # Fit the GP
    gp_model = pbo.fit_model_gp(dataset)
    print(f'[iter {i+beginNum+1}] {r.max().item()} bestsofar {bestSofar[-1]}')


bestSofar = [-1*i for i in bestSofar]
torch.save(bestSofar, directory_path+f'seed_{seed}.pt')
torch.save(rightNum, directory_path+f'RightNum{seed}.pt')