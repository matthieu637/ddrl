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
%p = zeros(1, discretization);
clear p
for i=1:discretization
  if(R(S(i)) >= R_absorbing)
      p{i}=[];
  else
  bestVal = -inf;
  indBestVal = A(1);
  p{i}=[A(1)];
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
      p{i}= [indBestVal];
    elseif(bestVal == possible_val)
      p{i}= [p{i} a];
    endif
  endfor
    %p(i) = indBestVal;
  endif
endfor

figure
for i=1:discretization
  plot(ones(1, size(p{i},2))*S(i), p{i},'.'); hold on;
endfor
title('pi *')
ylim([-1.1 1.1])

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

function samples = generateSample(samples, S_REQUIERED,S, A, R_absorbing, V, gamma)
  for k=1:S_REQUIERED
    is=randi(size(S,2));
    while ( R(S(is)) >= R_absorbing)
      is=randi(size(S,2));  
    endwhile
    
    ia=randi(size(A,2));
    
    next_s=T(S(is), A(ia));
    next_r=R(next_s);
    next_RVs = next_r;
    if(next_r < R_absorbing)
      [a_, closest] = min(abs(S - next_s));
      if(V(closest) == -Inf) 
        sub2= abs(S - next_s);
        sub2(closest)=Inf;
        [a_, closest] = min(sub2);
      endif
      next_RVs = next_RVs +gamma * V(closest);
    endif
    
    samples(end+1, :)=[is, ia, next_s, next_r, next_RVs, next_RVs];
  endfor

  %normalize Rvs
  a=min(samples(:,5));
  b=max(samples(:,5));
  samples(:,5)=(samples(:,5) - a)/(b-a);

endfunction

%compute Frontier
function y = kernel(s, s2, w )
  #gaussian
  y = gauss(s, s2, w)*(w/0.3989422804014327);
  
  #triangular
  #y = (1 - w *abs(s-s2));
  
  #epanechnikov
  #y = (1 - w *(s-s2)*(s-s2));
  
  #quadratic
  #y = (1 - w *(s-s2)*(s-s2))*(1 - w *(s-s2)*(s-s2));
endfunction

#400 gauss 0.12190
#400 trian  0.086440
#400 epanechnikov  0.097
#400 quadra 0.11710


S_REQUIERED=4;
samples=[];

old_sigma=zeros(1, discretization);
sigma=rand(1, discretization )/10;

samples = generateSample(samples, 400, S, A, R_absorbing, V, gamma);

h5=figure;
%while (sum(abs(old_sigma - sigma)) > 0.000001)
for pp=1:180

old_sigma = sigma;

%samples = generateSample(samples, S_REQUIERED, S, A, R_absorbing, V, gamma);


clear efficacity;
clear sample_efficacity;
clear phat;
Mrvs=zeros(1,size(S,2));
iMrvs=zeros(1,size(S,2));
 for i=1:size(S,2)
  clear local_rvs;
  for j=1:size(samples,1)
    local_rvs(end+1) = kernel(S(i), S(samples(j,1)), sigma(i))*samples(j,5);
  endfor
  [Mrvs(i), iMrvs(i)] = max(local_rvs);
  phat(i) = A(samples(iMrvs(i),2));
  if(size(p{i},2) == 0)
  efficacity(i) = 0;
  else
    efficacity(i) = min(abs(phat(i) - p{i}));
    
    
    %deducting over sigma
    %state is in the sample
    for j=find(samples (:,1) == i)'
      save_val = local_rvs(j);
      local_rvs(j)=-Inf;
      [idc, iMrvs_diff_than_me] = max(local_rvs);
      phat_diff_than_me = A(samples(iMrvs_diff_than_me,2));
      efficacity_diff_than_me = min(abs(phat_diff_than_me - p{i}));
    
      own_eff = min(abs(A(samples(j,2)) - p{i}));
      if(own_eff < efficacity(i) && iMrvs(i) != j) % i am better than the max without being it
%        printf('decrease sigma in %f \n', S(samples(j,1)));
        sigma(i) = sigma(i) + 0.01 * (own_eff - efficacity(i));
      elseif (own_eff > efficacity_diff_than_me && iMrvs(i) == j) %i am worst than the max(without me), but it's me
%        printf('increase sigma in %f \n', S(samples(j,1)));
        sigma(i) = sigma(i) + 0.01 * (own_eff - efficacity_diff_than_me)  ;
      else
        %printf('sigma ok\n');
      endif
      
       local_rvs(j) = save_val;
    endfor
    
    if(sum(samples (:,1) == i) >0)
      sample_efficacity(end+1)=efficacity(i);
    endif
  endif
 endfor
%plot(S,Mrvs, 'r')


%compute bestPolicy?
%figure(h1);
%plot(S, phat, 'r')

printf('eff : %f, sigma : %f, local eff : %f \n',mean(efficacity), sum(sigma), mean(sample_efficacity));
fflush(stdout);
figure(h5);
plot(sum(sigma), mean(efficacity), '.'); hold on;
axis()
%sleep(1);
%endwhile

if pp == 1
  h1=figure;
  for k=1:size(samples,1)
    is = samples(k,1);
    ia = samples(k,2);
    plot(S(is),A(ia)); hold on;
  endfor
  % for i=1:size(S,2)
  %  plot(S(i),A(samples(iMrvs(i),2)), 'ro');
  %endfor
  plot(S,A(samples(iMrvs,2)), 'r');

  title('SxA pi')
  axis([-1 1 -1 1])
endif

endfor


h9=figure;
for k=1:size(samples,1)
  is = samples(k,1);
  ia = samples(k,2);
  plot(S(is),A(ia)); hold on;
endfor
% for i=1:size(S,2)
%  plot(S(i),A(samples(iMrvs(i),2)), 'ro');
%endfor
plot(S,A(samples(iMrvs,2)), 'r');

title('SxA pi')
axis([-1 1 -1 1])


