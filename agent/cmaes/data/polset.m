#!/usr/bin/octave

%
% Extract different policies minimizing the number of wrong pol
%
close all; 
clear all; 

wanted_pol=50;
worst_perf=1500;

X=load_dirs ('.', '[0-9.]*learning.data', 6, 0, 0); plot(X,'.');
%Y=find(X < worst_perf); %all point greater than the worst perf
%Z=diff(Y); %consecutive diff

clear perf
for i=1:size(X,2)-wanted_pol
	perf(end+1)=sum(X(i:i+wanted_pol) < worst_perf);
endfor

[a,b]=max(perf)
figure
plot(X(b:b+wanted_pol),'.');
