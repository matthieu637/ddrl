close all;
clear all;


discretization=50;
gamma=0.9;

A=-1:2/(discretization-1):1;
S=-1:2/(discretization-1):1;

R_absorbing = 0;

function x=gauss(x,m,s)
  inv_sqrt_2pi = 0.3989422804014327;
  a=(x - m) / s;
  x = (inv_sqrt_2pi / s) * exp(-0.5 * a * a);
endfunction

function x = R_solo(s)
    x = -1 + gauss(s, 0.7, 0.05);
%    if(s < -0.3)
%      x = -1 + gauss(s, -0.3, 0.05);
%    else
%      x= -1;
%    endif
endfunction

function x = R(s)
  if size(s, 2) == 1
    x = R_solo(s);
  else
    x=zeros(size(s));
    for i=1:size(s, 2)
      x(i) = R_solo(s(i));
    endfor
  endif
endfunction

function x = T(s, a)
    x = s + a / 2.;
%    x1 = 0.5*s+ sin(a-0.5);
%    x2 = tanh(s * a + 0.2);
%    x3 = tanh(abs(s)-a);
%    x = (x1+x2 +x3);
%    xmin = -0.68461;
%    xmax= 1.8131;
%    x = ((2*(x - xmin))/(xmax-xmin)) -1;
endfunction

%return


% value iteration
V = zeros(1, discretization);
diffV = 1;
iter=0;
while diffV > 0.00001
  oldV=V;
  for i=1:size(V,2)
    bestVal = -inf;
    if R(S(i)) >= R_absorbing
      bestVal = -inf;
    else 
      for a=A
        sp=T(S(i), a);
        [_idc, sp_index] = min(abs(sp - S));
        possible_val = R(S(sp_index));
        if( R(S(sp_index))  < R_absorbing )
          possible_val = possible_val + gamma * V(sp_index);
        endif
        if(bestVal < possible_val)
          bestVal = possible_val;
        endif
      endfor
    endif
    V(i) = bestVal;
  endfor
  iter= iter + 1;
  diffV = sum(abs(oldV - V));
  if iter < 15
    diffV = 1;
  endif
endwhile
printf('value iteration : %d iter\n', iter);


% best policy
p = zeros(1, discretization);
for i=1:size(p,2)
  if(R(S(i)) >= R_absorbing)
      p(i)=-inf;
  else
  bestVal = -inf;
  indBestVal = A(1);
  for a=A
    sp=T(S(i), a);
    [_idc, sp_index] = min(abs(sp - S));
    possible_val = R(S(sp_index));
    if( R(S(sp_index))  < R_absorbing )
         possible_val = possible_val + gamma * V(sp_index);
    endif
    if(bestVal < possible_val)
      bestVal = possible_val;
      indBestVal = a;
    endif
  endfor
    p(i) = indBestVal;
  endif
endfor


%h1=figure;
%for k=1:S_REQUIERED
%  is = samples(k,1);
%  ia = samples(k,2);
%  plot(S(is),A(ia)); hold on;
%endfor
%title('SxA pi')
%axis([-1 1 -1 1])
%
%h2=figure;
%for k=1:S_REQUIERED
%  is = samples(k,1);
%  rvs = samples(k,5);
%  plot(S(is),rvs); hold on;
%endfor
%title('SxV score')
%axis([-1 1 0 1])

%compute Frontier
function y = kernel(s, s2, w )
  #gaussian
  #y = gauss(s, s2, w)*(w/0.3989422804014327);
  
  #triangular
  #y = (1 - w *abs(s-s2));
  
  #epanechnikov
  #y = (1 - w *(s-s2)*(s-s2));
  
  #quadratic
  y = (1 - w *(s-s2)*(s-s2))*(1 - w *(s-s2)*(s-s2));
endfunction

#400 gauss 0.12190
#400 trian  0.086440
#400 epanechnikov  0.097
#400 quadra 0.11710


for sigma=0.01:0.03:2

for mean_=1:10
S_REQUIERED=400;

samples=[];
for k=1:S_REQUIERED
  is=randi(size(S,2));
  ia=randi(size(A,2));
  
  next_s=T(S(is), A(ia));
  next_r=R(next_s);
  next_RVs = next_r;
  if(next_r < R_absorbing)
    [a_, closest] = min(abs(S - next_s));
    next_RVs = next_RVs +gamma * V(closest);
  endif
  
  samples(end+1, :)=[is, ia, next_s, next_r, next_RVs];
endfor

%normalize Rvs
a=min(samples(:,5));
b=max(samples(:,5));
samples(:,5)=(samples(:,5) - a)/(b-a);

clear Mrvs;
clear iMrvs;
clear efficacity;
clear phat;
 for i=1:size(S,2)
  clear local_rvs;
  for j=1:S_REQUIERED
    local_rvs(end+1) = kernel(S(i), S(samples(j,1)), sigma)*samples(j,5);
  endfor
  [Mrvs(end+1), iMrvs(end+1)] = max(local_rvs);
  phat(i) = A(samples(iMrvs(end),2));
  efficacity(i) = abs(phat(i)-p(i));
 endfor
%plot(S,Mrvs, 'r')


%compute bestPolicy?
%figure(h1);
%plot(S, phat, 'r')

MM(mean_)=mean(efficacity(find(efficacity != Inf)));
endfor

Eeff(end+1,:)=[mean(MM) sigma];
endfor
