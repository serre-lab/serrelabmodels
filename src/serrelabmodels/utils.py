import matplotlib.pyplot as plt
from matplotlib import rc
plt.style.use('ggplot')
import os
import numpy as np
import torch
import time
import pickle

def c_x(x, connectivity, reduce=True):
    '''c_x : weighted correlation between node feature vectors on a graph. Generalized from Eq. 2 in
        Brede, M. (2008). Synchrony-optimized networks of non-identical Kuramoto oscillators. Physics Letters, Section A: General,
        Atomic and Solid State Physics, 372(15), 2618–2622. https://doi.org/10.1016/j.physleta.2007.11.069
	
        Positional arguments are
	
        * x (tensor of shape n x d)            : the array of d-dimensional features for each of the n graph nodes. 
        * connectivity (tensor of shape n x n) : the graph's weight matrix. 
	
        Keyword arguments are
	
        * reduce (bool, default = True) : whether or not to average '''
	
	# Average value of the feature vector across nodes
    x_bar = x.mean(0).unsqueeze(0)
	
	# Weighted correlation
    num = (connectivity * torch.einsum('id,jd->ij',(x - x_bar) ,(x - x_bar) )).sum((0,1))
    den = (connectivity * ((x - x_bar)**2).unsqueeze(1).sum(-1)).sum((0,1))
    if reduce:
        return (num / den + 1e-6).mean()
    else:
        return (num / den + 1e-6)