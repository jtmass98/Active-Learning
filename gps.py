# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 08:21:51 2025

@author: jtm44
"""

import gpytorch
import numpy
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


class MyGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y):
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super().__init__(train_x, train_y,likelihood)
        self.N=train_x.shape[0]
        lengthscale_prior = gpytorch.priors.GammaPrior(2, 0.1)  # Adjust parameters as needed
        self.mean_module = gpytorch.means.ZeroMean()
        #matern lengthscale seems more stable with discrete inputs
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5, lengthscale_prior=lengthscale_prior)
        )
        self.likelihood=likelihood
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    def heatmap(self,N):
        #generate the current heatmap
        x=[[i,j] for i in range(N) for j in range(N)]
        self.eval()
        self.likelihood.eval()
        N=int(len(x)**0.5)
        test_y=self(torch.Tensor(x))#
        test_ys=np.zeros((N,N))
        for j,i in enumerate(x):
     
            test_ys[i[0],i[1]]=test_y[j].mean.item()
            
        return test_ys
            
            
            
    def update_train_data(self, new_train_x, new_train_y):
        #add the new point to the training set
        self.set_train_data(inputs=new_train_x, targets=new_train_y, strict=False)
        self.N=new_train_x.shape[0]
    
def EI(x,model,f_star):
    ##expected improvement function
    x=torch.Tensor(x.reshape(1,2))
    with torch.no_grad():  # Disable gradient computation
        out = model(x)
    mu,sig=out.mean.item(), out.stddev.item()
    I=max(0,mu-f_star)
    Z=I/max(sig,1e-9)
    ei = I*norm.cdf(Z) + 0.00000001*sig*norm.pdf(Z) #downweight the exploration
    return -ei 

def output(x,model):
    #used to find the maximum temperature
    x=torch.Tensor(x.reshape(1,2))
    with torch.no_grad():  # Disable gradient computation
        out = model(x)
    mu,sig=out.mean.item(), out.stddev.item()

    return -mu ##negative so it is minimised to find the maximum
        
        
        
        
        

def next_sample(data,model,size):
    #find the coordinates of the next sample to measure
    train_x=data[0].reshape(-1,2)
    train_y=data[1].reshape(-1,)
    f_star=torch.max(train_y)
    
    model=train_gps(model,train_x, train_y)
    
    
    
    model.eval()
    model.likelihood.eval()
    next_sample=optimiser(model,f_star,size)
    maxi=maximiser(model,size)
    return next_sample,maxi


def train_gps(model,train_x,train_y):
    model.train()
    model.likelihood.train()
    # Optimizer (Adam)
    lr=1
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Marginal Log Likelihood (MLL) loss
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    num_iterations = 100000
    iterator = (tqdm(range(num_iterations), desc="Epoch"))
      # Number of training iterations
    history=[]
    best_loss=10000000
    count=0
    for i in iterator:
        optimizer.zero_grad()  # Zero gradients
        output = model(train_x)  # Forward pass
        loss = -mll(output,train_y)  # Compute negative MLL
        loss.backward()  # Backpropagation
        optimizer.step()  # Update parameters
        
        
        ####stopping criteria
        current_loss=np.mean(history[-20:])
        if current_loss<best_loss-0.01:
            count=0
            best_loss=current_loss
        elif best_loss+0.01<current_loss:
            break
        elif best_loss-0.01<current_loss:
            count+=1
        if count>1000 and lr>0.001:
            optimizer.param_groups[0]['lr']*=0.1
            count=0
        if count>1000 and lr<=0.005:
            break
        lr = optimizer.param_groups[0]['lr'] 
        history.append(loss.item())
        iterator.set_postfix(loss=[np.round(loss.item(),3)])
        
    return model

def optimiser(model,f_star,size):
    ##find the next spot to measure
    bounds = [(0, size),(0,size)]  # Search space bounds
    x_init = np.random.rand(200, 2)*size  # Random restarts
    
    best_x = None
    best_ei = float("inf")
    
    for x0 in x_init:
        res = minimize(EI, x0, args=(model, f_star), bounds=bounds, method="L-BFGS-B")
    
        if res.fun < best_ei:
            best_ei = res.fun
            best_x = res.x
    return best_x
    
def maximiser(model,size):
    #find the current maximum
    bounds = [(0, size),(0,size)]  # Search space bounds
    x_init = np.random.rand(200, 2)*size  # Random restarts
    
    best_x = None
    best_ei = float("inf")
    
    for x0 in x_init:
        res = minimize(output, x0, args=(model), bounds=bounds, method="L-BFGS-B")
        
        if res.fun < best_ei:
            best_ei = res.fun
            best_x = res.x
    return best_x







