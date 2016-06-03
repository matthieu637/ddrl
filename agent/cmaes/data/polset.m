#!/usr/bin/octave

%
% Extract different policies minimizing the number of wrong pol
%


%minimizing score
%maximazing the number of perf != than the worst
close all; 
clear all; 

wanted_pol=50;
worst_perf=1500;

X=load_dirs ('.', '[0-9.]*learning.data', 6, 0, 0); plot(X,'.');

clear perf
for i=1:size(X,2)-wanted_pol
	perf(end+1)=sum(X(i:i+wanted_pol) < worst_perf);
endfor

[a,b]=max(perf)
figure
plot(X(b:b+wanted_pol),'.');


%with stochasticity and maximizing the score
%maximazing the variance of perf > worst
close all; 
clear all; 

wanted_pol=50;
worst_perf=80;
X=load_dirs ('..', '[0-9.]*testing.data', 6, 0, 0); plot(median(X),'.');
Xsave= X;
X=median(X);

clear perf
for i=1:size(X,2)-wanted_pol
	subX=X(i:i+wanted_pol);
	%subX(find(subX < worst_perf))=mean(find(subX > worst_perf));%penalize variance for worst perf
	subX(find(subX < worst_perf))=[];%penalize variance for worst perf

	%distance from all point to another
	mysum=0;
	for j=1:size(subX,2)
		mysum = mysum +	sum(abs(subX(j)-subX));
	endfor
        perf(end+1)=mysum;

	%perf(end+1)=sum(abs(diff(sort(subX))));
endfor

perf(1:31)=0;

[a,b]=max(perf)
figure
plot(X(b:b+wanted_pol),'.');


