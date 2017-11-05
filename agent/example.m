close all;
clear all;


discretization=200;
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

figure 
Tdis = zeros(discretization, discretization);
for i=1:size(Tdis,1)
  for j=1:size(Tdis,2)
    Tdis(i,j) = T(S(i), A(j));
  endfor
endfor
plot3(S,A, Tdis);
title('T')
xlabel('s')
ylabel('a')
axis([ -1 1 -1 1 -1 1])

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
   if iter < 15
    diffQ = 1;
  endif
endwhile
printf('value iteration : %d iter\n', iter);

figure
xplot=-1:2/discretization:1;
plot(xplot, R(xplot))
title('R')


figure
plot(S, V, '.')
title('V *')


figure
for i=1:size(S,2)
  bestVal = -inf;
  for j=1:size(A,2)
    possible_val = Q(i, j);
    if(bestVal < possible_val)
      bestVal = possible_val;
    endif
  endfor
  VfromQ(i)=bestVal;
endfor
plot(S, VfromQ)
title('V * from Q')


figure
QfromV = zeros(discretization, discretization);
for i=1:size(QfromV,1)
  for j=1:size(QfromV,2)
    if(R(S(i)) >= R_absorbing)
        QfromV(i,j) = -inf;
    else
      sp=T(S(i), A(j));
      [_idc, sp_index] = min(abs(sp - S));
      possible_val = R(S(sp_index));
      if( R(S(sp_index))  < R_absorbing )
        possible_val = possible_val + gamma * V(sp_index);
      endif
      QfromV(i,j) = possible_val;
    endif
  endfor
endfor
plot3(S, A, QfromV)
title('Q * from V')
xlabel('s')
ylabel('a')


figure
plot(S, p)
title('pi *')
ylim([-1.1 1.1])


figure
[xx,yy] = meshgrid (linspace (-1,1,discretization));
griddata(S, A, Q, xx, yy, "linear");
%plot3(S, A, Q, '.')
title('Q *')
xlabel('s')
ylabel('a')




