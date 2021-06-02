import matplotlib.pyplot as plt
import os
import seaborn
from scipy import stats
import numpy as np


def plot_samples(samples, log_interval, unrolled_steps, path):
    
    cmap = 'Blues'
    bbox=[-2, 2, -2, 2]
    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    cols = len(samples)
    fig, ax = plt.subplots(nrows=1, ncols=cols, figsize=(8 * cols, 8), sharex=True, sharey=True)
    
    for i, samps in enumerate(samples):
            
        kernel = stats.gaussian_kde(samps.T)
        f = np.reshape(kernel(positions).T, xx.shape)
        
        if cols == 1:
            cfset = ax.contourf(xx, yy, f, cmap=cmap)
            ax.axis(bbox)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            cfset = ax[i].contourf(xx, yy, f, cmap=cmap)
            ax[i].axis(bbox)
            ax[i].set_aspect('equal')
            ax[i].set_xticks([])
            ax[i].set_yticks([])

    plt.gcf().tight_layout()
    plt.savefig(path)
    #plt.show()
    plt.close()