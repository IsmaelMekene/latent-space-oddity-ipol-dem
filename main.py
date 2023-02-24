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




# Makes a Dx1 vector given x
def my_vector(x):
    return np.asarray(x).reshape(-1, 1)



# Synthetic datasets
def generate_data(params=None):

    # The semi-circle data in 2D
    if params['data_type'] == 1:
        N = params['N']
        theta = np.pi * np.random.rand(N, 1)
        data = np.concatenate((np.cos(theta), np.sin(theta)), axis=1) + params['sigma'] * np.random.randn(N, 2)
        return data

    # A simple 2-dim surface in 3D with a hole in the middle
    elif params['data_type'] == 2:
        N = params['N']
        Z = np.random.rand(N, 2)
        Z = Z - np.mean(Z, 0).reshape(1, -1)
        Centers = np.zeros((1, 2))
        dists = pairwise_distances(Z, Centers)  # The sqrt(|x|)
        inds_rem = (dists <= params['r']).sum(axis=1)  # N x 1, The points within the ball
        Z_ = Z[inds_rem == 0, :]  # Keep the points OUTSIDE of the ball
        F = (np.sin(2 * np.pi * Z_[:, 0])).reshape(-1, 1)
        F = F + params['sigma'] * np.random.randn(F.shape[0], 1)
        data = np.concatenate((Z_, 0.25 * F), axis=1)
        return data

    # Two moons on a surface and with extra noisy dimensions
    elif params['data_type'] == 3:
        N_all = params['N']
        N = int(N_all / 2)
        theta = np.pi * np.random.rand(N, 1)
        z1 = np.concatenate((np.cos(theta), np.sin(theta)), axis=1)
        z2 = np.concatenate((np.cos(theta), -np.sin(theta)), axis=1) + my_vector([1.0, 0.25]).T
        z = np.concatenate((z1, z2), axis=0) + params['sigma'] * np.random.randn(int(N * 2), 2)
        z = z - z.mean(0).reshape(1, -1)
        z3 = (np.sin(np.pi * z[:, 0])).reshape(-1, 1)
        z3 = z3 + params['sigma'] * np.random.randn(z3.shape[0], 1)
        data = np.concatenate((z, 0.5 * z3), axis=1)
        if params['extra_dims'] > 0:
            noise = params['sigma'] * np.random.randn(N_all, params['extra_dims'])
            data = np.concatenate((data, noise), axis=1)
        labels = np.concatenate((0 * np.ones((z1.shape[0], 1)), np.ones((z2.shape[0], 1))), axis=0)
        return data, labels

    return -1






#params = {'N': 200, 'data_type': 3, 'sigma': 0.1, 'extra_dims': 2, 'r':1}
#data, labels = utils.generate_data(params)
#data, labels = generate_data(params)

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
