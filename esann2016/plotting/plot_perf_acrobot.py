from pylab import *
from palettable.colorbrewer.qualitative import Set2_7


 # brewer2mpl.get_map args: set name  set type  number of colors
colors = Set2_7.mpl_colors

 
params = {
    'axes.labelsize': 8,
    'text.fontsize': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [4, 4]
}
rcParams.update(params)


def load(f):
    return np.loadtxt(f)

def perc(data):
    median = np.zeros(data.shape[1])
    perc_25 = np.zeros(data.shape[1])
    perc_75 = np.zeros(data.shape[1])
    for i in range(0, len(median)):
        median[i] = np.median(data[:, i])
        perc_25[i] = np.percentile(data[:, i], 25)
        perc_75[i] = np.percentile(data[:, i], 75)
    return median, perc_25, perc_75

data_rand = load('../result_data/adacrobot/cl-off.perf.data')
data_human = load('../result_data/adacrobot/cl-on.perf.data') 
data_bootstrap = load('../result_data/adacrobot/random.perf.data')
#data_power = load('power.data')


n_generations = min(data_human.shape[1], data_rand.shape[1] )
x=np.arange(0, n_generations)
lx= np.arange(0, n_generations, 1)

discre=1
x_data_bootstrap = np.arange(0, data_bootstrap.shape[1], discre)
lx_data_bootstrap = np.arange(0, data_bootstrap.shape[1], discre)

med_rand, perc_25_rand, perc_75_rand = perc(data_rand)
med_human, perc_25_human, perc_75_human = perc(data_human)
med_bootstrap, perc_25_bootstrap, perc_75_bootstrap = perc(data_bootstrap)
#med_power, perc_25_power, perc_75_power = perc(data_power)

fig = figure() # no frame
ax = fig.add_subplot(111)

ax.set_title('Median and quartile of different algorithm')

# now all plot function should be applied to ax
ax.fill_between((lx), perc_25_rand[x], perc_75_rand[x], alpha=0.35, linewidth=0, color=colors[0]) 
ax.fill_between((lx), perc_25_human[x], perc_75_human[x], alpha=0.25, linewidth=0, color=colors[1])
#ax.fill_between(lx, perc_25_power[lx], perc_75_power[lx], alpha=0.25, linewidth=0, color=colors[3])
ax.fill_between(x_data_bootstrap, perc_25_bootstrap[x_data_bootstrap], perc_75_bootstrap[x_data_bootstrap], alpha=0.15, linewidth=0, color=colors[2])

ax.plot((lx), med_rand[lx], linewidth=2, linestyle=':', color=colors[0])
ax.plot((lx), med_human[lx], linewidth=2, linestyle='--', color=colors[1])
#ax.plot(lx, med_power[lx], linewidth=2, linestyle='-.', color=colors[3])
ax.plot(lx_data_bootstrap, med_bootstrap[lx_data_bootstrap], linewidth=2, linestyle='-', color=colors[2])

# change xlim to set_xlim

#ax.invert_yaxis()
ax.set_ylim(250, 502)
#ax.set_xscale('log')
#ax.set_yscale('log')
#ax.set_xlim(128, 1e6)
ax.set_xlim(0, 1000)

#change xticks to set_xticks
#ax.set_xticks(np.arange(0, 10001, 2500))
#ax.set_yticks(np.arange(0, 25, 4))

legend = ax.legend(["NFAC", "cacla" , "random"], loc=3);
legend.set_frame_on(False)
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('1.0')

# put the grid behind
ax.set_axisbelow(True)

ax.set_xlabel("episode")
ax.set_ylabel("step")

fig.savefig('../result_plotting/adacrobot-1ddl_perf.png', dpi=200)

