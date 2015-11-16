X_on = load_dirs('../data/cartpole/cacla-on/', 'learning.data', 6, 1, 1);
X_off = load_dirs('../data/cartpole/cacla-off/', 'learning.data', 6,1, 1);
X_rand = load_dirs('../data/cartpole/random/', 'learning.data', 6, 1, 1);
X_cma = load_dirs('../data/cartpole/cmaes/', 'learning.data', 6, 1, 1);

save cartpole/cl-on.perf.data X_on
save cartpole/cl-off.perf.data X_off
save cartpole/random.perf.data X_rand
save cartpole/cmaes.perf.data X_cma

