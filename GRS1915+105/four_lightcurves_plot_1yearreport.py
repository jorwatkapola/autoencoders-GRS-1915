lc_classes = np.array(lc_classes)
import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = (15.0, 10.0)
alpha = np.where(lc_classes == "alpha")[0][3]
rho = np.where(lc_classes == "rho")[0][0]
lambda1 = np.where(lc_classes == "lambda")[0][0]
theta = np.where(lc_classes == "theta")[0][1]


import matplotlib.ticker as ticker

fig, axes = plt.subplots(nrows=2, ncols=2)

plt.subplots_adjust(hspace=0)
plt.subplots_adjust(wspace=0)

lc_len=20000
axes[0][0].plot(lcs[alpha][0][:lc_len]-lcs[alpha][0][0],lcs[alpha][1][:lc_len], c="green", linewidth=0.5, zorder=-5)
axes[0][1].plot(lcs[rho][0][:lc_len]-lcs[rho][0][0],lcs[rho][1][:lc_len], c="green", linewidth=0.5, zorder=-5)
axes[1][0].plot(lcs[lambda1][0][-lc_len:]-lcs[lambda1][0][-lc_len],lcs[lambda1][1][-lc_len:], c="green", linewidth=0.5, zorder=-5)
axes[1][1].plot(lcs[theta][0][3200:lc_len+3200]-lcs[theta][0][3200], lcs[theta][1][3200:lc_len+3200], c="green", linewidth=0.5, zorder=-5)

axes[0][0].set_ylim([0, 11000])
axes[0][1].set_ylim([0, 11000])
axes[1][0].set_ylim([0, 11000])
axes[1][1].set_ylim([0, 11000])

axes[0][0].set_xlim([0, 2500])
axes[0][1].set_xlim([0, 2500])
axes[1][0].set_xlim([0, 2500])
axes[1][1].set_xlim([0, 2500])

axes[0][0].tick_params(axis="x", which="major", length=5, width=1, labelsize=5, direction="in")
axes[0][1].tick_params(axis="x", which="major", length=5, width=1, labelsize=5, direction="in")
axes[1][0].tick_params(axis="x", which="major", length=5, width=1, labelsize=25, direction="in")
axes[1][1].tick_params(axis="x", which="major", length=5, width=1, labelsize=25, direction="in")

axes[0][0].tick_params(axis="y", which="major", length=5, width=1, labelsize=25, direction="in")
axes[1][0].tick_params(axis="y", which="major", length=5, width=1, labelsize=25, direction="in")
axes[0][1].tick_params(axis="y", which="major", length=5, width=1, labelsize=5, direction="in")
axes[1][1].tick_params(axis="y", which="major", length=5, width=1, labelsize=5, direction="in")

axes[0][0].text(1,1,"(a)", ha='right', va='top', transform=axes[0][0].transAxes, size=30)
axes[0][1].text(1,1,"(b)", ha='right', va='top', transform=axes[0][1].transAxes, size=30)
axes[1][0].text(1,1,"(c)", ha='right', va='top', transform=axes[1][0].transAxes, size=30)
axes[1][1].text(1,1,"(d)", ha='right', va='top', transform=axes[1][1].transAxes, size=30)

plt.setp(axes[0][1].get_yticklabels(), visible=False)
plt.setp(axes[1][1].get_yticklabels(), visible=False)
plt.setp(axes[0][1].get_xticklabels(), visible=False)
plt.setp(axes[0][0].get_xticklabels(), visible=False)

axes[1][0].set_xticks([500,1000,1500,2000])
axes[1][1].set_xticks([500,1000,1500,2000])

axes[0][0].set_ylabel("counts /sec", size=30)
axes[1][0].set_xlabel("time (sec)", size=30)

axes[0][0].yaxis.set_label_coords(-0.2, 0)
axes[1][0].xaxis.set_label_coords(1, -0.1)

plt.savefig('sample_lightcurves_12Mreport.png', dpi=300)
fig.show()