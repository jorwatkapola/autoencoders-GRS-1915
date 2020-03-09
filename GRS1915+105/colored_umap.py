# load observation classifications from Huppenkothen 2017

clean_belloni = open('../../1915Belloniclass_updated.dat')
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
with open('../../data/95921_len512_s40_counts.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../../data/95921_len512_s40_errors.pkl', 'rb') as f:
    errors = pickle.load(f)
with open('../../data/95921_len512_s40_ids.pkl', 'rb') as f:
    seg_ids = pickle.load(f)

# HF QPO observation ids
paper_obIDs = np.loadtxt("Belloni_Altamirano_obsIDs.txt", dtype=str)

qpo_colours = []

for seg_id in seg_ids:
    if seg_id.split("_")[0] in paper_obIDs:
        qpo_colours.append("red")
    else:
        qpo_colours.append("grey")
        
qpo_labels = []

for seg_id in seg_ids:
    if seg_id.split("_")[0] in paper_obIDs:
        qpo_labels.append("QPO")
    else:
        qpo_labels.append("other")
        
        
qpo_scales = []

for seg_id in seg_ids:
    if seg_id.split("_")[0] in paper_obIDs:
        qpo_scales.append("QPO")
    else:
        qpo_scales.append("other")
        
        
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
cm.get_cmap(plt.get_cmap("Set1"))


colours = ['#ffd8b1', '#000075', '#808080', '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#000000']

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15.0, 15.0)
plt.rcParams.update({'font.size': 22})

# fig, ax = plt.subplots()
class_indices = np.where(np.array(class_name) == "Unknown")[0]
class_data = embeddings_lap[class_indices]

plt.scatter(class_data[:,0], class_data[:,1], s = 0.2, c="grey", label="Unknown")

for plot_class_ind, plot_class in enumerate(classes):
    class_indices = np.where(np.array(class_name) == plot_class)[0]
    class_data = embeddings_lap[class_indices]
    
    plt.scatter(class_data[:,0], class_data[:,1], s = 25, c=colours[plot_class_ind], label=plot_class)
    
# plt.legend()
plt.title("UMAP embedding of the encoded GRS1915 segments, neighbors=50, min_dist=0.0, components=2", fontsize=12)


redint = np.where(np.array(qpo_colours) == "red")
greyint= np.where(np.array(qpo_colours) != "red")
plt.scatter(embeddings_lap[:,0][greyint], embeddings_lap[:,1][greyint], s=1, c="grey", label= "other")
plt.scatter(embeddings_lap[:,0][redint], embeddings_lap[:,1][redint], s=1, c="red", label= "HF QPO")
plt.title("UMAP embedding of the encoded GRS1915 segments, neighbors=50, min_dist=0.0, components=2", fontsize=12)
plt.legend()
plt.savefig("Belloni_Altamirano_2013_embedded.png")
plt.show()