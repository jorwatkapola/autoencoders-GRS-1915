import pickle
import numpy as np


with open('../../../data_GRS1915/94465_len512_s40_counts.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../../../data_GRS1915/94465_len512_s40_errors.pkl', 'rb') as f:
    errors = pickle.load(f)
with open('../../../data_GRS1915/94465_len512_s40_ids.pkl', 'rb') as f:
    seg_ids = pickle.load(f)
embeddings = np.loadtxt("../../../model_2020-02-09_10-36-06_embeddings_94465.csv", delimiter=",")

with open('../../../data_GRS1915/model_2020-02-09_10-36-06_UMAP_mapper_94465.pkl', 'rb') as f:
    UMAP_mapper = pickle.load(f)
    
    
umapembed = UMAP_mapper.transform(embeddings)


# load observation classifications from Huppenkothen 2017
%matplotlib inline

import matplotlib.pyplot as plt


clean_belloni = open('../../../data_GRS1915/1915Belloniclass_updated.dat')
lines = clean_belloni.readlines()
states = lines[0].split()
belloni_clean = {}
for h,l in zip(states, lines[1:]):
    belloni_clean[h] = l.split()
    #state: obsID1, obsID2...
ob_state = {}
for state, obs in belloni_clean.items():
    if state == "chi1" or state == "chi2" or state == "chi3" or state == "chi4": state = "chi"
    for ob in obs:
        ob_state[ob] = state
        
# load segmented light curves

import pickle
# with open('../../../data_GRS1915/95921_len512_s40_counts.pkl', 'rb') as f:
#     segments = pickle.load(f)
# with open('../../../data_GRS1915/95921_len512_s40_errors.pkl', 'rb') as f:
#     errors = pickle.load(f)
# with open('../../../data_GRS1915/95921_len512_s40_ids.pkl', 'rb') as f:
#     seg_ids = pickle.load(f)






# # HF QPO observation ids
# paper_obIDs = np.loadtxt("../../../data_GRS1915/Belloni_Altamirano_obsIDs.txt", dtype=str)

# qpo_colours = []

# for seg_id in seg_ids:
#     if seg_id.split("_")[0] in paper_obIDs:
#         qpo_colours.append("red")
#     else:
#         qpo_colours.append("grey")
        
# qpo_labels = []

# for seg_id in seg_ids:
#     if seg_id.split("_")[0] in paper_obIDs:
#         qpo_labels.append("QPO")
#     else:
#         qpo_labels.append("other")
        
        
# qpo_scales = []

# for seg_id in seg_ids:
#     if seg_id.split("_")[0] in paper_obIDs:
#         qpo_scales.append("QPO")
#     else:
#         qpo_scales.append("other")
        
        
xxx = [seg.split("_")[0] for seg in seg_ids]

classes = np.array(["alpha", "beta", "gamma", "delta", "theta", "kappa", "lambda", "mu", "nu", "rho", "phi", "chi", "eta", "omega"])
class_colour = []
for ob in xxx:
    if ob in ob_state:
        class_colour.append(np.where(classes == ob_state[ob])[0][0])
    else:
        class_colour.append(15)
        
classes = np.array(["alpha", "beta", "gamma", "delta", "theta", "kappa", "lambda", "mu", "nu", "rho", "phi", "chi", "eta", "omega"])
scales = []
class_name = []
for ob in xxx:
    if ob in ob_state:
        class_name.append(ob_state[ob])
        scales.append(5)
    else:
        class_name.append("Unknown")
        scales.append(0.1)
        
        
from matplotlib import cm
# cm.get_cmap(plt.get_cmap("Set1"))


colours = ['#ffd8b1', '#000075', '#808080', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#000000']

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# sns.set_style("white")
plt.rcParams['figure.figsize'] = (30.0, 30.0)
plt.rcParams.update({'font.size': 0})

embeddings_lap = umapembed

# fig, ax = plt.subplots()

fig, axs = plt.subplots(1, 1)
# axs = axs.flatten()

class_indices = np.where(np.array(class_name) == "Unknown")[0]
class_data = embeddings_lap[class_indices]
axs.scatter(class_data[:,0], class_data[:,1], s = 1, c="grey", label="Unknown")

for plot_class_ind, plot_class in enumerate(classes):


    class_indices = np.where(np.array(class_name) == plot_class)[0]
    class_data = embeddings_lap[class_indices]
    
    axs.scatter(class_data[:,0], class_data[:,1], s = 5, c='red', label=plot_class)
    
# plt.legend()
#     axs[plot_class_ind].set_title("{}".format(plot_class), fontsize=42)
# axs.reshape((4,4))
# plt.savefig("classes_separate.png")
plt.title("Classified segments in red, unclassified in grey", fontsize=42)

plt.savefig("figures/UMAP_embedding_all_class_red_s5.png")
plt.show()


# redint = np.where(np.array(qpo_colours) == "red")
# greyint= np.where(np.array(qpo_colours) != "red")
# plt.scatter(embeddings_lap[:,0][greyint], embeddings_lap[:,1][greyint], s=1, c="grey", label= "other")
# plt.scatter(embeddings_lap[:,0][redint], embeddings_lap[:,1][redint], s=1, c="red", label= "HF QPO")
# plt.legend()
plt.show()