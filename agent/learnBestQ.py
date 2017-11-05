from fann2 import libfann
import oct2py
import numpy as np
from pylab import *
from palettable.colorbrewer.qualitative import Set2_7
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


HIDDEN=70
# 5 OR 3 for pi(s)
LEARN='best_Q'

print('hidden unit', HIDDEN)

# brewer2mpl.get_map args: set name  set type  number of colors
colors = Set2_7.mpl_colors
close('all')

oc = oct2py.Oct2Py();
Y = oc.load(LEARN).Q;

X = oc.load('best_X').S[0];
oc.exit()

#2 to learn
X2 = [];
Y2 = [];

#3-4 to plot
X3 = [];
Y3 = [];

for i in range(0, X.size):
	for j in range(0, X.size):
		if Y[i][j] != -float('inf'):
			X2.append([X[i], X[j]])
			Y2.append([Y[i][j]])
		X3.append([X[i], X[j]])

X2 = np.array(X2);
Y2 = np.array(Y2);

train_data = libfann.training_data();
train_data.set_train_data(X2, Y2);

def learn():
	ann = libfann.neural_net();
	ann.create_standard_array([2, HIDDEN, 1]);
	ann.set_training_algorithm(libfann.TRAIN_RPROP);
	ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
	ann.set_activation_function_output(libfann.LINEAR)
	ann.train_on_data(train_data, 4000, 0, 0.005);
	return (ann, ann.get_MSE());

bestScore = float('inf');
ann = [];
for i in range(0, 5):
	(nn, score) = learn();
	print('score : ', score);
	if score < bestScore:
		ann = nn;
		bestScore = score

print('best score : ', bestScore);

YC=[]
for i in range(0, X2.shape[0]):
	yc_ = ann.run(X2[i]);
	YC.append(yc_[0])


fig = figure() # no frame
ax = fig.gca(projection='3d');

#ax.scatter(X2[:, 0], X2[:, 1], Y2[:, 0]);
#ax.plot_trisurf(X2[:, 0], X2[:, 1], Y2[:, 0], cmap=cm.jet, linewidth=0.2);


#ax.set_ylim(-3, 8);
#ax.plot(X4, YC, linewidth=3, linestyle='-', color=colors[1])

YC=np.array(YC);
ax.plot_trisurf(X2[:, 0], X2[:, 1], YC, cmap=cm.jet, linewidth=0.2);


