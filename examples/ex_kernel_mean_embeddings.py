import numpy as np
import pandas as pd
import seaborn as sns

# Toy example taken from
# https://www.inference.vc/causal-inference-2-illustrating-interventions-in-a-toy-example/
N = 10000;                # sample size
sigma = 1;             # RBF kernel bandwidth

# Producing joint distributions with different underlying causal strctures:
# x -> y
d1_xy= np.zeros((N,2));
for i in range(0,N-1):
    d1_xy[i,0] = np.random.randn(); # x
    d1_xy[i,1] = 1 + d1_xy[i,0] + np.sqrt(3)*np.random.randn(); # y

df1 = pd.DataFrame(d1_xy, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df1);

#  x <- y
d2_xy= np.zeros((N,2));
for i in range(0,N-1):
    d2_xy[i,1] = 1 + 2*np.random.randn(); # y
    d2_xy[i,0] = 1 + (d2_xy[i,1]-1)/4 + np.sqrt(3)*np.random.randn()/2; # x

df1 = pd.DataFrame(d1_xy, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df1);

# x <- z -> y
d3_xy= np.zeros((N,2));
for i in range(0,N-1):
    z = np.random.randn();
    d3_xy[i,1] = z + 1 + np.sqrt(3)*np.random.randn(); # y
    d3_xy[i,0] = z; # x

df3 = pd.DataFrame(d3_xy, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df3);

# The joint distributions are indistinguishable
from sklearn.metrics.pairwise import rbf_kernel

# Radial basis kernel functions
k1 = rbf_kernel(d1_xy, d1_xy, 1/pow(sigma,2));
k2 = rbf_kernel(d2_xy, d2_xy, 1/pow(sigma,2));
k12 = rbf_kernel(d1_xy, d2_xy, 1/pow(sigma,2));

# Kernel Two Sample test (Estimating the maximum mean discrepancy)
# https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions
mmd_12 = 1/pow(N,2)*sum(map(sum, k1)) + 1/pow(N,2)*sum(map(sum, k2)) - 2/pow(N,2)*sum(map(sum, k12));
