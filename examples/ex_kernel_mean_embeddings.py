import numpy as np
import pandas as pd
import seaborn as sns

# Toy example taken from
# https://www.inference.vc/causal-inference-2-illustrating-interventions-in-a-toy-example/
N = 1000

# Producing indistinguishable joint distributions
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
