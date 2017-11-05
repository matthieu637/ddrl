close all; clear all; 

X=1:100; 

ymax=60;

Y=rand([1 size(X,2)])*(2*ymax)-ymax; 
Y=randn([1 size(X,2)])*ymax;

plot(X,Y, '.')
for i=1:size(X,2)
  for j=1:size(X,2)
    if i != j
      D(end+1)= abs(Y(i) - Y(j));
    endif
  endfor
endfor

figure; plot(1:size(D,2),D, '.'); ylim([0 ymax*2]);

figure; plot(1:size(D,2), D/(sqrt(var(D))/0.26115), '.');

