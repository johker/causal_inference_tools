import numpy as np
import pandas as pd
import seaborn as sns

from scipy.optimize import fsolve

# Calculate p(y|x) considered causal direction
samples = 10000
x = np.array([-1.1,0.3,1.1])
p_y_given_x = np.zeros((samples, len(x)))

# Test data: y = x³ + x + epsilon
for i, xi in enumerate(x):
    p_y_given_x[:,i] = pow(xi,3) + xi + np.random.normal(0,1,samples)

# https://seaborn.pydata.org/tutorial/distributions.html
sns.kdeplot(p_y_given_x[:,0], shade=True)
sns.kdeplot(p_y_given_x[:,1], shade=True)
sns.kdeplot(p_y_given_x[:,2], shade=True)

# Calculate p(x|y) considered anticausal direction
y = np.array([-1.1,0.3,1.1])
p_x_given_y = np.zeros((samples,len(y)))
x_guess = 1
epsilon = np.random.normal(0,1,samples)

# We are using a numerical approximation
for i, yi in enumerate(y):
    for j,epsj in enumerate(epsilon):
        func = lambda x : yi - x**3 - x - epsj
        x_guess = 1 if(j==0) else p_x_given_y[j-1,i]
        p_x_given_y[j,i] = fsolve(func, p_x_given_y[j,i])

# The representation is more complex (higher structural variability)
np.mean(p_x_given_y[:,1])
p_x_given_y[:,1].shape
sns.kdeplot(p_x_given_y[:,0], shade=True)
sns.kdeplot(p_x_given_y[:,1], shade=True)
sns.kdeplot(p_x_given_y[:,2], shade=True)
