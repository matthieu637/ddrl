
close all;
clear all;

discretization=75;
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

clear V

% value iteration Q
Q = zeros(discretization, discretization);
diffQ = 1;
iter=0;
while diffQ > 0.0001
  oldQ=Q;
  for i=1:size(Q,1)
    for j=1:size(Q,2)
      if(R(S(i)) >= R_absorbing)
        Q(i,j) = -inf;
      else 
        sp=T(S(i), A(j));
        [_idc, sp_index] = min(abs(sp - S));
        Q(i,j) = R(S(sp_index)) ;
        if( R(S(sp_index))  < R_absorbing )
          Q(i,j) = Q(i,j)  + gamma * max(Q(sp_index, :));
        endif
      endif
    endfor
  endfor
  iter= iter + 1;
  diffQ = sum(sum(abs(oldQ - Q)));
   if iter < 25
    diffQ = 1;
  endif
endwhile
printf('value iteration : %d iter\n', iter);


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

function samples = generateSample(samples, S_REQUIERED,S, A, R_absorbing, Q, gamma)
  for k=1:S_REQUIERED
    is=randi(size(S,2));
    while ( R(S(is)) >= R_absorbing)
      is=randi(size(S,2));  
    endwhile
    
    ia=randi(size(A,2));
    next_RVs = Q(is,ia);
    
    samples(end+1, :)=[is, ia, 0, next_RVs];
  endfor

  %normalize Rvs
  a=min(samples(:,4));
  b=max(samples(:,4));
  samples(:,3)=(samples(:,4) - a)/(b-a);

endfunction

function samples = generateSample2(samples, S_REQUIERED,S, A, R_absorbing, Q, gamma,iMrvs)
  for k=1:S_REQUIERED
    is=randi(size(S,2));
    while ( R(S(is)) >= R_absorbing)
      is=randi(size(S,2));  
    endwhile
    
    %ia=samples(iMrvs(is),2)+randi(7)-4;
    ia=samples(iMrvs(is),2)+int64(randn(1)*size(A,2)/3);
    ia=max(ia, 1);
    ia=min(ia, size(A,2));
    next_RVs = Q(is,ia);
    
    samples(end+1, :)=[is, ia, 0, next_RVs];
  endfor

  %normalize Rvs
  a=min(samples(:,4));
  b=max(samples(:,4));
  samples(:,3)=(samples(:,4) - a)/(b-a);

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

function eff= efficacite_sigma(samples, sigma, S, Q, p, A)
  efficacity=zeros(1,size(S,2));
  phat=zeros(1,size(S,2));
  Mrvs=zeros(1,size(S,2));
  iMrvs=zeros(1,size(S,2));
   %for i=1:size(S,2)
   for i=randperm(size(S,2))
    clear local_rvs;
    for j=1:size(samples,1)
      local_rvs(end+1) = kernel(S(i), S(samples(j,1)), sigma)*samples(j,3);
    endfor
    [Mrvs(i), iMrvs(i)] = max(local_rvs);
    phat(i) = A(samples(iMrvs(i),2));
    if(size(p{i},2) == 0)
    efficacity(i) = 0;
    else
      efficacity(i) = min(abs(phat(i) - p{i}));
      current_eff = Q(i,samples(iMrvs(i),2));

      if(sum(samples (:,1) == i) >0)
        sample_efficacity(end+1)=efficacity(i);
      endif
    endif
   endfor
  eff= mean(efficacity);

endfunction

#400 gauss 0.12190
#400 trian  0.086440
#400 epanechnikov  0.097
#400 quadra 0.11710


S_REQUIERED=4;
samples=[];

sigma=rand;
sigma_init=sigma;

%samples = generateSample(samples, 70, S, A, R_absorbing, Q,gamma);
samples = generateSample(samples, 5, S, A, R_absorbing, Q,gamma);

h5=figure
%while (abs(old_sigma - sigma) > 0.000001)
%for pp=1:500
for pp=1:0

  old_sigma = sigma;

  %if( mod(pp, 10) == 0)
    %samples = generateSample2(samples, S_REQUIERED, S, A, R_absorbing, Q, gamma,iMrvs);
  %endif
    
  want_increase=[];
  want_decrease=[];
  efficacity=zeros(1,size(S,2));
  clear sample_efficacity;
  phat=zeros(1,size(S,2));
  Mrvs=zeros(1,size(S,2));
  iMrvs=zeros(1,size(S,2));
   %for i=1:size(S,2)
   for i=randperm(size(S,2))
    clear local_rvs;
    for j=1:size(samples,1)
      local_rvs(end+1) = kernel(S(i), S(samples(j,1)), sigma)*samples(j,3);
    endfor
    [Mrvs(i), iMrvs(i)] = max(local_rvs);
    phat(i) = A(samples(iMrvs(i),2));
    if(size(p{i},2) == 0)
    efficacity(i) = 0;
    else
      efficacity(i) = min(abs(phat(i) - p{i}));
      current_eff = Q(i,samples(iMrvs(i),2));
      
      %deducting over sigma
      %state is in the sample
      for j=find(samples (:,1) == i)'
        save_val = local_rvs;
        local_rvs(j)=-Inf;
        local_rvs(find(samples(:,1) == samples(j,1)))=-Inf; %same s
        local_rvs(find(samples(:,3) <= samples(j,3) +0.001))=-Inf; %lower rvs
        
        [idc, iMrvs_diff_than_me] = max(local_rvs);
        %phat_diff_than_me = A(samples(iMrvs_diff_than_me,2));
        %efficacity_diff_than_me = min(abs(phat_diff_than_me - p{i}));
        efficacity_diff_than_me = Q(i, samples(iMrvs_diff_than_me,2));
        
        %own_eff = min(abs(A(samples(j,2)) - p{i}));
        own_eff = Q(samples(j,1), samples(j,2));
        % i am better than the max without being it & the max is not on the same s & i should be the higher rvs in this state
        if(own_eff > current_eff && iMrvs(i) != j && samples(iMrvs(i) ,1) != samples(j,1) ) && samples(j,3) > 0
          wanted_sigma = ifelse(samples(iMrvs(i),3) > samples(j,3), log(samples(iMrvs(i),3)/samples(j,3)) , log(samples(j,3)/samples(iMrvs(i),3)));
          dist = (S(samples(iMrvs(i),1)) - S(samples(j,1)));
          dist = dist * dist;
          wanted_sigma = sqrt(dist/(2 * wanted_sigma));
  %        printf('decrease sigma in %f ->  (wanted) %f \n', S(samples(j,1)), wanted_sigma);
          want_decrease(end+1)=wanted_sigma;
          sigma = sigma + 0.01 * (wanted_sigma - sigma);

        %i am worst than the max(without me), but it's me +diff state + high rvs
        elseif (own_eff < efficacity_diff_than_me && iMrvs(i) == j) 
          wanted_sigma = ifelse(samples(iMrvs_diff_than_me,3) > samples(j,3), log(samples(iMrvs_diff_than_me,3)/samples(j,3)), log(samples(j,3)/samples(iMrvs_diff_than_me,3)));
          dist = (S(samples(iMrvs_diff_than_me,1)) - S(samples(j,1)));
          dist = dist * dist;
          wanted_sigma = sqrt(dist/(2 * wanted_sigma));
  %        printf('increase sigma in %f -> (wanted) %f \n', S(samples(j,1)), wanted_sigma);
          sigma = sigma + 0.01 * (wanted_sigma - sigma);
          want_increase(end+1)=wanted_sigma;
        else
          %printf('sigma ok\n');
        endif
        
         local_rvs = save_val;
      endfor
      
      if(sum(samples (:,1) == i) >0)
        sample_efficacity(end+1)=efficacity(i);
      endif
      
  %    if(size(want_increase, 2) > 0 && size(want_decrease, 2) > 0 )
  %      sigma = ((max(want_increase) - min(want_decrease))/2) ;
  %    elseif (size(want_increase, 2) > 0)
  %      sigma = max(want_increase) + 0.01;
  %    elseif (size(want_decrease, 2) > 0)
  %      sigma = min(want_decrease) - 0.01;
  %    endif
    endif
   endfor
  %plot(S,Mrvs, 'r')


  %compute bestPolicy?
  %figure(h1);
  %plot(S, phat, 'r')

  printf('eff : %f, sigma : %f, local eff : %f \n',mean(efficacity), sigma, mean(sample_efficacity));
  fflush(stdout);
  figure(h5);
  plot(sigma, mean(efficacity), '.'); hold on;



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


h8=figure;
for k=1:size(samples,1)
  is = samples(k,1);
  ia = samples(k,2);
  plot(S(is),A(ia)); hold on;
endfor
% for i=1:size(S,2)
%  plot(S(i),A(samples(iMrvs(i),2)), 'ro');
%endfor

%plot(S,A(samples(iMrvs,2)), 'ro--');

%title('SxA pi')
%axis([-1 1 -1 1])




%
%
%
%
%
%


sigma=sigma_init

for pp=1:0

  %if( mod(pp, 10) == 0)
    %samples = generateSample2(samples, S_REQUIERED, S, A, R_absorbing, Q, gamma,iMrvs);
  %endif
    
  clear want_increase;
  clear want_decrease;
  efficacity=zeros(1,size(S,2));
  clear sample_efficacity;
  phat=zeros(1,size(S,2));
  Mrvs=zeros(1,size(S,2));
  iMrvs=zeros(1,size(S,2));
   %for i=1:size(S,2)
   for i=randperm(size(S,2))
    clear local_rvs;
    for j=1:size(samples,1)
      local_rvs(end+1) = kernel(S(i), S(samples(j,1)), sigma)*samples(j,3);
    endfor
    [Mrvs(i), iMrvs(i)] = max(local_rvs);
    phat(i) = A(samples(iMrvs(i),2));
    if(size(p{i},2) == 0)
    efficacity(i) = 0;
    else
      efficacity(i) = min(abs(phat(i) - p{i}));
      current_eff = Q(i,samples(iMrvs(i),2));
      
      %deducting over sigma
      %state is in the sample
      for j=find(samples (:,1) == i)'
        save_val = local_rvs;
        local_rvs(j)=-Inf;
        local_rvs(find(samples(:,1) == samples(j,1)))=-Inf; %same s
        local_rvs(find(samples(:,3) <= samples(j,3) +0.001))=-Inf; %lower rvs
        
        [idc, iMrvs_diff_than_me] = max(local_rvs);
        %phat_diff_than_me = A(samples(iMrvs_diff_than_me,2));
        %efficacity_diff_than_me = min(abs(phat_diff_than_me - p{i}));
        efficacity_diff_than_me = Q(i, samples(iMrvs_diff_than_me,2));
        
        %own_eff = min(abs(A(samples(j,2)) - p{i}));
        own_eff = Q(samples(j,1), samples(j,2));
        % i am better than the max without being it & the max is not on the same s & i should be the higher rvs in this state
        if(own_eff > current_eff && iMrvs(i) != j && samples(iMrvs(i) ,1) != samples(j,1) ) && samples(j,3) > 0
          wanted_sigma = ifelse(samples(iMrvs(i),3) > samples(j,3), log(samples(iMrvs(i),3)/samples(j,3)) , log(samples(j,3)/samples(iMrvs(i),3)));
          dist = (S(samples(iMrvs(i),1)) - S(samples(j,1)));
          dist = dist * dist;
          wanted_sigma = sqrt(dist/(2 * wanted_sigma));
%          printf('decrease sigma in %f ->  (wanted) %f \n', S(samples(j,1)), wanted_sigma);
          want_decrease(end+1,:)=[wanted_sigma S(i)];

        %i am worst than the max(without me), but it's me +diff state + high rvs
        elseif (own_eff < efficacity_diff_than_me && iMrvs(i) == j) 
          wanted_sigma = ifelse(samples(iMrvs_diff_than_me,3) > samples(j,3), log(samples(iMrvs_diff_than_me,3)/samples(j,3)), log(samples(j,3)/samples(iMrvs_diff_than_me,3)));
          dist = (S(samples(iMrvs_diff_than_me,1)) - S(samples(j,1)));
          dist = dist * dist;
          wanted_sigma = sqrt(dist/(2 * wanted_sigma));
%          printf('increase sigma in %f -> (wanted) %f \n', S(samples(j,1)), wanted_sigma);
          want_increase(end+1,:)=[wanted_sigma S(i)];
        else
          %printf('sigma ok\n');
        endif
        
         local_rvs = save_val;
      endfor
      
      if(sum(samples (:,1) == i) >0)
        sample_efficacity(end+1)=efficacity(i);
      endif
      
%      if(size(want_increase, 2) > 0 && size(want_decrease, 2) > 0 )
%        sigma = ((max(want_increase) - min(want_decrease))/2) ;
%      elseif (size(want_increase, 2) > 0)
%        sigma = max(want_increase);
%      elseif (size(want_decrease, 2) > 0)
%        sigma = min(want_decrease);
%      endif
    endif
   endfor
  %plot(S,Mrvs, 'r')


  %compute bestPolicy?
  %figure(h1);
  %plot(S, phat, 'r')

  printf('eff : %f, sigma : %f, local eff : %f \n',mean(efficacity), sigma, mean(sample_efficacity));
  fflush(stdout);
  figure(h5);
  plot(sigma, mean(efficacity), 'r.'); hold on;

endfor

figure
for sigma=0.01:0.01:1
  plot(sigma, efficacite_sigma(samples, sigma, S, Q, p, A), '.');hold on;
endfor

if false

figure
for i=want_decrease'
  plot([i(2) i(2)],[i(1) 0], 'r'); hold on;
endfor

for i=want_increase'
  plot([i(2) i(2)],[i(1) 1]);
endfor

endif

[a,b]=max(Q(samples(:,1), samples(:,2))'); % perfect result of sigma


A=(S(samples(:,1))'-S(samples(:,1))).^2;

B=-2*log(Q(samples(:,1),samples(:,2))./Q(samples(:,1),samples(:,2))');

Sig = real(sqrt(A./B)) ;%all possible sigma?

for i=1:size(b,2)
  h(end+1)=Sig(i, b(i));
  f(end+1)=Sig(b(i), i);
endfor
f=f(find(f>0));
h=h(find(h>0));
mean(f)
mean(h)


