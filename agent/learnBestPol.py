from fann2 import libfann
import oct2py
import numpy as np
from pylab import *
from palettable.colorbrewer.qualitative import Set2_7

HIDDEN=10
# 5 OR 3 for pi(s)
LEARN='best_policy'
LEARN='best_valuef'

print('hidden unit', HIDDEN)

# brewer2mpl.get_map args: set name  set type  number of colors
colors = Set2_7.mpl_colors
close('all')

oc = oct2py.Oct2Py();
if LEARN=='best_policy':
	Y = oc.load(LEARN).p[0];
else:
	Y = oc.load(LEARN).V[0];

X = oc.load('best_X').S[0];
oc.exit()

X2 = [];
Y2 = [];
X3 = [];
X4 = [];
Y3 = [];
for i in range(0, X.size):
	if Y[i] != -float('inf'):
		X2.append([X[i]])
		Y2.append([Y[i]])
		X4.append(X[i])
		Y3.append(Y[i])
	else:
		Y3.append(np.nan)
	X3.append(X[i])

X2 = np.array(X2);
Y2 = np.array(Y2);

train_data = libfann.training_data();
train_data.set_train_data(X2, Y2);

def learn():
	ann = libfann.neural_net();
	ann.create_standard_array([1, HIDDEN, 1]);
	ann.set_training_algorithm(libfann.TRAIN_RPROP);
	ann.set_activation_function_hidden(libfann.SIGMOID_SYMMETRIC)
	if LEARN=='best_policy':
		ann.set_activation_function_output(libfann.SIGMOID_SYMMETRIC)
	else:
		ann.set_activation_function_output(libfann.LINEAR)
	ann.train_on_data(train_data, 15000, 0, 0.00005);
	return (ann, ann.get_MSE());

bestScore = float('inf');
ann = [];
for i in range(0, 15):
	(nn, score) = learn();
	print('score : ', score);
	if score < bestScore:
		ann = nn;
		bestScore = score

print('best score : ', bestScore);

YC=[]
for i in range(0, X2.size):
	yc_ = ann.run(X2[i]);
	YC.append(yc_[0])


fig = figure() # no frame
ax = fig.add_subplot(111)
ax.plot(X3, Y3, '--', color=colors[0])

if LEARN=='best_policy':
	ax.set_ylim(-1.1, 1.1);
else:
	ax.set_ylim(-3, 8);

ax.plot(X4, YC, linewidth=3, linestyle='-', color=colors[1])
#show()
