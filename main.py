import os
import sys 
import numpy as np
import sklearn_extra
from sklearn_extra.cluster import KMedoids

import torch
import torch.nn as nn
from python.core import utils, generative_models, manifolds, geodesics, 
import matplotlib.pyplot as plt
import gc
from sklearn.cluster import KMeans

import numpy as np


from sklearn.metrics import pairwise_distances

import plotly.graph_objects as go

from tqdm import tqdm








#params = {'N': 200, 'data_type': 3, 'sigma': 0.1, 'extra_dims': 2, 'r':1}
#data, labels = utils.generate_data(params)
#z = data[:, 2]
#x, y = data[:,0], data[:,1]
#fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z,
                                   #mode='markers', marker=dict(
        #size=4,
        #color=z,                # set color to an array/list of desired values
        #colorscale='Viridis',   # choose a colorscale
        #opacity=0.8
    #))])
#fig.show()

#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))


MU_Z_data = np.load(os.path.join(ROOT, 'MU_Z_data2.npy'))
plt.figure()
plt.scatter(MU_Z_data[:, 0], MU_Z_data[:, 1], c='k', s=15)
plt.title('Latent space')



affinity = np.load('/content/affinity2.npy')


def apply_kmedioids (data, metric = 'euclidean',affinity = None ):
    
    if (metric == 'euclidean'):
        kmedio =  KMedoids(n_clusters=2)
        kmedio.fit(data)


# labels 
        label = kmedio.labels_

# plot
        c = ['b','y']
        for l in np.unique(label):
            plt.scatter(data[label == l][:,0], data[label == l][:,1], label = str(l))

        #plt.scatter(kmedio.cluster_centers_[:,0], kmedio.cluster_centers_[:,1], marker = '*', label = 'centroids' , s = 200)

        plt.legend()
        
    else : 
        kmedio_rienman =  KMedoids(n_clusters=2,max_iter = 100000, metric = 'precomputed' )
        kmedio_rienman.fit(affinity)


# labels 
        label = kmedio_rienman.labels_

# plot
        c = ['b','y']
        for l in np.unique(label):
            plt.scatter(data[label == l][:,0], data[label == l][:,1], label = str(l))

        medioid_indices = kmedio_rienman.medoid_indices_

        #plt.scatter(data[medioid_indices,0], data[medioid_indices,1], marker = '*', label = 'centroids' , s = 200)

        plt.legend()
    return label 
        
    
    
sigma = 1.6
affinity_kernel = np.exp (- affinity / sigma**2)
predicted_labels_riemannian = apply_kmedioids (MU_Z_data, metric = 'Riemann', affinity = affinity_kernel)
plt.title ('Riemannian Kmediods')


predicted_labels_euclidean = apply_kmedioids (MU_Z_data, metric = 'euclidean')
plt.title ('Euclidean Kmediods')
















'''

# if you need to access a file next to the source code, use the variable ROOT
# for example:
#    torch.load(os.path.join(ROOT, 'weights.pth'))
ROOT = os.path.dirname(os.path.realpath(__file__))

def main(input, output, sigma):
    u = iio.read(input)
    print("hello world", u.shape)
    
    v = u + np.random.randn(*u.shape) * sigma

    iio.write(output, v)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sigma", type=float, required=True)
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()
    main(args.input, args.output, args.sigma)
    
'''
