with open('../../../data_GRS1915/1776_light_curves_1s_bin_errorfix.pkl', 'rb') as f:
    lcs = pickle.load(f)
with open('../../../data_GRS1915/1776_light_curves_1s_bin_ids_errorfix.pkl', 'rb') as f:
    ids = pickle.load(f)
    
    
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

        

inv_ob_state = {}
for k, v in ob_state.items():
    inv_ob_state[v] = inv_ob_state.get(v, [])
    inv_ob_state[v].append(k)


import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
#https://stackoverflow.com/questions/8389636/creating-over-20-unique-legend-colors-using-matplotlib


pylab.rcParams['figure.figsize'] = (29.7, 21.0) # A4 size 210mm x 297mm

colors = matplotlib.colors.CSS4_COLORS.keys()
colors = np.array(list(colors))


NUM_COLORS = 14
cm = plt.get_cmap("jet")#('gist_rainbow')
colors = [cm(1.*i/NUM_COLORS) for i in range(NUM_COLORS)]

ids_ar = np.array(ids)

class_names = list(inv_ob_state.keys())


alpha = lcs[np.where(ids_ar == inv_ob_state["alpha"][0])[0][0]]
beta= lcs[np.where(ids_ar == inv_ob_state["beta"][5])[0][0]] #3
gamma=lcs[np.where(ids_ar == inv_ob_state["gamma"][0])[0][0]]
delta=lcs[np.where(ids_ar == inv_ob_state["delta"][9])[0][0]]
theta=lcs[np.where(ids_ar == inv_ob_state["theta"][13])[0][0]]#11
kappa=lcs[np.where(ids_ar == inv_ob_state["kappa"][6])[0][0]]#6
lambda1=lcs[np.where(ids_ar == inv_ob_state["lambda"][3])[0][0]] #3
mu=lcs[np.where(ids_ar == inv_ob_state["mu"][6])[0][0]]#6
nu=lcs[np.where(ids_ar == inv_ob_state["nu"][0])[0][0]]
rho=lcs[np.where(ids_ar == inv_ob_state["rho"][9])[0][0]]#9
phi=lcs[np.where(ids_ar == inv_ob_state["phi"][3])[0][0]]# 3,6
chi=lcs[np.where(ids_ar == inv_ob_state["chi"][27])[0][0]]# 1,17,27
eta=lcs[np.where(ids_ar == inv_ob_state["eta"][2])[0][0]]# 1
# omega=lcs[np.where(ids_ar == inv_ob_state["kappa"][-3])[0][0]]
omega=lcs[np.where(ids_ar == inv_ob_state["omega"][1])[0][0]]


selected_lcs = [alpha,beta,gamma,delta,theta,kappa,lambda1,mu,nu,rho,phi,chi,eta,omega]


fig, axes = plt.subplots(nrows=7, ncols=2)
axes = axes.flatten()

plt.subplots_adjust(hspace=0.05)
plt.subplots_adjust(wspace=0.01)

good_classes = ["delta", "mu", "rho", "phi"]
intervals = {}

for plot_ind in range(14):
    light_c = np.copy(selected_lcs[plot_ind])
    class_name = class_names[plot_ind]
    offset = light_c[0][0]
    axes[plot_ind].set_ylim([-0.1, 1.1])
    
    if class_name == "alpha":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1
        start=0
        end =breaks[0]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][:breaks[0]]-offset, light_c[1][:breaks[0]], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].set_xlim([0, 3500])
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)

    elif class_name == "beta":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start=breaks[0]
        end =breaks[1]
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "gamma":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start=breaks[0]
        end =breaks[1]
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "theta":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start=breaks[1]
        end =breaks[2]
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "kappa":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start=breaks[-1]
        end =-1
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "lambda":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start=breaks[-1]
        end =-1
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "nu":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start= breaks[0]
        end =breaks[1]
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "chi":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start= breaks[0]
        end =-1
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "eta":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start= breaks[2]
        end =breaks[3]
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "omega":
        breaks = np.where((light_c[0][1:]-light_c[0][:-1]) != 1.)[0]+1 # [ 279 3584 6652]
        start= breaks[0]
        end =-1
        offset = light_c[0][start]
        light_c[1] /= np.max(light_c[1][start:end])*1.1
        axes[plot_ind].plot(light_c[0][start:end]-offset, light_c[1][start:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].plot(light_c[0][start:end-48]-offset+1607, light_c[1][start+48:end], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
    elif class_name == "rho":
        light_c[1] /= np.max(light_c[1])*1.1
        axes[plot_ind].plot(light_c[0]-offset, light_c[1], c=(colors[plot_ind][0]*0.95, colors[plot_ind][1]*0.95, colors[plot_ind][2]*0.95, 1.0), linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
        
        
    elif class_name in good_classes:
        
        light_c[1] /= np.max(light_c[1])*1.1
        axes[plot_ind].plot(light_c[0]-offset, light_c[1], c=colors[plot_ind], linewidth=1, zorder=-5)
        axes[plot_ind].text(1,1,r"$\{}$".format(class_name), ha='right', va='top', transform=axes[plot_ind].transAxes, size=30)
        axes[plot_ind].set_xlim([0, 3500])
    else:
        axes[plot_ind].plot(light_c[0]-offset, light_c[1])
        axes[plot_ind].plot(light_c[0][:3500]-offset, light_c[1][:3500])
    
    axes[plot_ind].set_xlim([0, 2500])
#     axes[plot_ind].tick_params(axis="x", which="major", length=5, width=1, labelsize=20, direction="in")
    
    if plot_ind%2 == 0:
        axes[plot_ind].tick_params(axis="y", which="major", length=5, width=1, labelsize=20, direction="in")
    else:
        axes[plot_ind].tick_params(axis="y", which="major", length=5, width=1, labelsize=0, direction="in")
        plt.setp(axes[plot_ind].get_yticklabels(), visible=False)
    if plot_ind == 6:
        axes[plot_ind].set_ylabel("Brightness of the black hole", size=30)
    if plot_ind == 12 or plot_ind == 13:
        axes[plot_ind].tick_params(axis="x", which="major", length=5, width=1, labelsize=20, direction="in")
        axes[plot_ind].set_xlabel("Time in seconds", size=30)

    else:
        axes[plot_ind].tick_params(axis="x", which="major", length=5, width=1, labelsize=0, direction="in")
        plt.setp(axes[plot_ind].get_xticklabels(), visible=False)
    
    axes[plot_ind].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[plot_ind].set_yticklabels(["dim","","","", "bright"])
    axes[plot_ind].set_xticks([0, 500, 1000,1500, 2000])
    axes[plot_ind].set_xticklabels([0,"",1000,"", 2000])

    
axes.reshape((7,2))


# axes[0][0].tick_params(axis="x", which="major", length=5, width=1, labelsize=5, direction="in")
# axes[0][1].tick_params(axis="x", which="major", length=5, width=1, labelsize=5, direction="in")
# axes[1][0].tick_params(axis="x", which="major", length=5, width=1, labelsize=25, direction="in")
# axes[1][1].tick_params(axis="x", which="major", length=5, width=1, labelsize=25, direction="in")

# axes[0][0].tick_params(axis="y", which="major", length=5, width=1, labelsize=25, direction="in")
# axes[1][0].tick_params(axis="y", which="major", length=5, width=1, labelsize=25, direction="in")
# axes[0][1].tick_params(axis="y", which="major", length=5, width=1, labelsize=5, direction="in")
# axes[1][1].tick_params(axis="y", which="major", length=5, width=1, labelsize=5, direction="in")

# plt.setp(axes[0][1].get_yticklabels(), visible=False)
# plt.setp(axes[1][1].get_yticklabels(), visible=False)
# plt.setp(axes[0][1].get_xticklabels(), visible=False)
# plt.setp(axes[0][0].get_xticklabels(), visible=False)

# axes[1][0].set_xticks([500,1000,1500,2000])
# axes[1][1].set_xticks([500,1000,1500,2000])


# axes[0][0].yaxis.set_label_coords(-0.2, 0)
# axes[1][0].xaxis.set_label_coords(1, -0.1)

# plt.suptitle("14 classes of activity of x-ray black hole binary GRS1915+105", fontsize=40, y=0.92)


plt.savefig('all_classes_of_GRS1915.png', dpi=300)

fig.show()