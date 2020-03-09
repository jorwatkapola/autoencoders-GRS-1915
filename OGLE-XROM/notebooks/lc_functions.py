from IPython.display import clear_output
from segment_cluster import segmentation
import importlib
import os
import fnmatch
import numpy as np
from collections import Counter
import csv
from sklearn import tree
import sys
sys.stdout.flush()
import math
import pickle
from scipy.stats import zscore
import datetime
import pytz
import matplotlib.pyplot as plt


def load_and_bin():
    """
    load light curve data from txt files, bin the data to 1 second resolution and return a list of light curves and ids
    
    """
    lcs=[]
    ids=[]

    for root, dirnames, filenames in os.walk("/home/jakub/Documents/GRS1915+105/data/Std1_PCU2"):
        for filename in fnmatch.filter(filenames, "*_std1_lc.txt"):
            lc = os.path.join(root, filename)
            ids.append(filename.split("_")[0])
            f=np.loadtxt(lc)
            f=np.transpose(f)#,axis=1)
            #f=f[0:2]
            ###1s average and time check to eliminate points outside of GTIs
            f8t = np.mean(f[0][:(len(f[0])//8)*8].reshape(-1, 8), axis=1)
            f8c = np.mean(f[1][:(len(f[1])//8)*8].reshape(-1, 8), axis=1)
            f8e = np.mean(f[2][:(len(f[1])//8)*8].reshape(-1, 8), axis=1)
            rm_points = []
            skip=False
            for i in range(len(f8t)-1):
                if skip==True:
                    skip=False
                    continue
                delta = f8t[i+1]-f8t[i]
                if delta > 1.0:
                    rm_points.append(i+1)
                    skip=True

            times=np.delete(f8t,rm_points)
            counts=np.delete(f8c,rm_points)
            errors=np.delete(f8e,rm_points)
            lcs.append(np.stack((times,counts, errors)))
    return lcs, ids