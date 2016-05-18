#!/usr/bin/octave

%
% Extract different policies minimizing the number of wrong pol
%

wanted_pol=50;
worst_perf=1500;

close all; clear all; X=load_dirs ('.', '[0-9.]*learning.data', 6, 0, 0); plot(X,'.');
Xsave=X;
X=diff(find(X < worst_perf));

clear perf
for i=1:size(X,2)-wanted_pol
	perf(end+1)=sum(X(i:i+wanted_pol));
endfor

[a,b]=min(perf)
figure
plot(X(b:b+wanted_pol),'.');
