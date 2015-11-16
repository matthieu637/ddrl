X_on = load_dirs('../data/acrobot-1ddl/cacla-on/', 'learning.data', 4, 1, 1);
X_off = load_dirs('../data/acrobot-1ddl/cacla-off/', 'learning.data', 4,1, 1);
X_rand = load_dirs('../data/acrobot-1ddl/random/', 'learning.data', 4, 1, 1);


save adacrobot/cl-on.goal.data X_on
save adacrobot/cl-off.goal.data X_off
save adacrobot/random.goal.data X_rand

X_on = load_dirs('../data/acrobot-1ddl/cacla-on/', 'learning.data', 6, 1, 0);
X_off = load_dirs('../data/acrobot-1ddl/cacla-off/', 'learning.data', 6,1, 0);
X_rand = load_dirs('../data/acrobot-1ddl/random/', 'learning.data', 6, 1, 0);
X_cma = load_dirs('../data/acrobot-1ddl/cmaes/', 'learning.data', 6, 1, 0);

save adacrobot/cl-on.perf.data X_on
save adacrobot/cl-off.perf.data X_off
save adacrobot/random.perf.data X_rand
save adacrobot/cmaes.perf.data X_cma

%plotMedianQ(X_on, 'r')
%plotMedianQ(X_off, 'y')
%plotMedianQ(X_ql, 'g')
%plotMedianQ(X_power, 'g')

%xlim([0,200])
%xlim([0,100])
