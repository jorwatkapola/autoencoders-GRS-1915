import pickle
with open('../../../data_GRS1915/94465_len512_s40_counts.pkl', 'rb') as f:
    segments = pickle.load(f)
with open('../../../data_GRS1915/94465_len512_s40_errors.pkl', 'rb') as f:
    errors = pickle.load(f)
with open('../../../data_GRS1915/94465_len512_s40_ids.pkl', 'rb') as f:
    seg_ids = pickle.load(f)

with open('../../../data_GRS1915/1776_light_curves_1s_bin.pkl', 'rb') as f:
    lcs = pickle.load(f)
with open('../../../data_GRS1915/1776_light_curves_1s_bin_ids.pkl', 'rb') as f:
    ids = pickle.load(f)
    
    
import numpy as np
embeddings = np.loadtxt("../../../model_2020-02-09_10-36-06_embeddings_94465.csv", delimiter=",")


with open('../../../data_GRS1915/model_2020-02-09_10-36-06_UMAP_mapper_94465.pkl', 'rb') as f:
    UMAP_mapper = pickle.load(f)

umapembed = UMAP_mapper.transform(embeddings)
embeddings_lap = umapembed

start_times = []
for lc in lcs:
    start_times.append(lc[0][0])
    
ordered_ob_ids = np.array(ids)[np.argsort(np.array(start_times))]

import pandas as pd

ids_df = pd.DataFrame({"seg_ids": seg_ids, "ob_id": [seg.split("_")[0] for seg in seg_ids], "index": np.array([seg.split("_")[1] for seg in seg_ids], dtype=int)})

sorted_segments = []
for ob in ordered_ob_ids:
    sorted_segments = np.hstack((sorted_segments, np.array(ids_df[ids_df["ob_id"] == ob].sort_values("index").seg_ids)))
    
chrono_latent_coords = []
for curr_seg in sorted_segments:
    curr_ind = np.where(np.array(seg_ids) == curr_seg)[0][0]
    chrono_latent_coords.append(embeddings_lap[curr_ind])
    
chrono_latent_coords = np.asarray(chrono_latent_coords)

rev_chrono_latent_coords = np.flip(chrono_latent_coords, axis=0)



#https://towardsdatascience.com/animations-with-matplotlib-d96375c5442c
#https://stackoverflow.com/questions/9401658/how-to-animate-a-scatter-plot
import matplotlib.pyplot as plt 
import matplotlib.animation as animation 
import numpy as np 
# plt.rcParams.update(plt.rcParamsDefault)

plt.rcParams['figure.figsize'] = (10.0, 10.0)
plt.rcParams.update({'font.size': 0})

fig, ax = plt.subplots()
ax.set_xlim([-8.5, 8.5])

ax.set_ylim([-8.5, 8.5])

# ([-8.5, 8.5, -8.5, 8.5]) #xlim=(-8.5, 8.5), ylim=(-8.5, 8.5)

x = -9
y = -9
# initialization function 
text = ax.text(0.50, 0.05, s=str("------"), size=20, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)
# mark = ax.scatter(x, y, s = 90, c="cyan", zorder=10, marker="x") 
scat = ax.scatter(rev_chrono_latent_coords[:,0], rev_chrono_latent_coords[:,1], s = sizes_array, c="blue", zorder=2) 
# line = ax.plot(rev_chrono_latent_coords[:,0], rev_chrono_latent_coords[:,1], lw = 2, c="red", zorder=5)
line, = ax.plot([], [], lw=2, c="red", zorder=5)
xdata, ydata = [], [] 

sizes_array = np.array([0]*94465)
colour_array = np.array(["grey"]*94465)
x=-8.5
y=-8.5

def init(): 

    ax.scatter(rev_chrono_latent_coords[:,0], rev_chrono_latent_coords[:,1], s = 1, c="grey")
    line.set_data([], [])
#     mark = ax.scatter(x, y, s = 90, c="blue", zorder=10, marker="x") 
    scat = ax.scatter(rev_chrono_latent_coords[:,0], rev_chrono_latent_coords[:,1], s = sizes_array, c="blue", zorder=2) 
    
    return line, text, scat




# animation function 
def animate(i):
    index_1 = 94465-1-i*15
    index_2 = 94465-1-(i-1)*15
    xs = rev_chrono_latent_coords[index_1:index_2][:,0]
    ys = rev_chrono_latent_coords[index_1:index_2][:,1]
#     xdata = np.hstack((xdata,xs)) 
#     ydata = np.hstack((ydata,ys)) 
    xdata = rev_chrono_latent_coords[index_1:index_2+50][:,0]
    ydata = rev_chrono_latent_coords[index_1:index_2+50][:,1]
    line.set_data(xdata, ydata)
    
    if index_1 < 0: index_1 = 0
        
    sizes_array[index_1:index_2] = 5
#     colour_array[index_1:index_2] = "red"
    
#     x, y = rev_chrono_latent_coords[index_1-1]
#     xdata.append()
#     ydata.append()
    
#     text = ax.text(0.90, 0.05, s=str(rev_sorted_segments[index_1-1]), size=30, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes)

#     rev_sorted_segments_ob_ids

    text.set_text("{}/94465   ob_id: ".format(str(i*15)) +str(rev_sorted_segments_ob_ids[index_1-1]))
    scat.set_sizes(sizes_array)
#     mark.set_offsets((x,y))
#     scat.set_array(colour_array)
    
    return line, text, scat

# setting a title for the plot 
plt.title('GRS1915+105 light curve segments (512sec) in latent space, appearing chronologically', fontsize=12) 
# hiding the axis details 
# plt.axis('off') 

# call the animator	 
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=6298, interval=200, blit=True) #94465#6298


# anim = animation.FuncAnimation(fig, animate, init_func=init, frames=6298, interval=34, blit=True) #94465#6298


# save the animation as mp4 video file 
anim.save('GRS1915_latent_6298frames_200ms.gif',writer='imagemagick') 