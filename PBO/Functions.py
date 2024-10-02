import math
import numpy as np
import torch
import random
import ObjFunc

# All functions are defined in such a way that have global maximums,
# if a function originally has a minimum, the final objective value is multiplied by -1

class TestFunction:
    def evaluate(self,x):
        pass

    def getOptimal(self):
        return self.optimal


class Rosenbrock(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-5, 10]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.rosenbrock_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f


class Griewank(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-50, 50]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.griewank_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f


class Sphere(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-5.12, 5.12]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.sphere_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f


class Levy(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-10., 10.]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.levy_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f
    
   
class Schwefel(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-50, 50]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.Schwefel_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f
    
class Dixon(TestFunction):
    def __init__(self, objDim, effDim, noise_var=0):
        random.seed(0)
        self.rand_list = random.sample([dim for dim in range(objDim)], effDim)
        self.range = np.array([[-10, 10]] * objDim)
        self.var = noise_var
        self.optimal = 0

    def scale_domain(self, x):
        # Scaling the domain
        x_copy = np.copy(x)
        if len(x_copy.shape) == 1:
            x_copy = x_copy.reshape((1, x_copy.shape[0]))
        for i in range(len(self.range)):
            x_copy[:, i] = x_copy[:, i] * (self.range[i, 1] - self.range[i, 0]) / 2 + (
                    self.range[i, 1] + self.range[i, 0]) / 2
        return x_copy

    def evaluate_true(self, x):
        # Calculating the output
        scaled_x = self.scale_domain(x)
        x = torch.from_numpy(scaled_x)
        f = ObjFunc.Dixon_rand(x, self.rand_list).reshape(1, -1).numpy()
        f = np.transpose(f)
        return f
    
