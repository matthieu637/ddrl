clear all;
close all;

%debug_on_error(1)

function load_P(ep) 
  X=load(strcat("P.",num2str(ep)));
  if ( exist(strcat("aP",num2str(ep))) == 0)
    Y = [];
  elseif ((stat(strcat("aP",num2str(ep))).size) == 0)
    Y=[];
  else
    Y=load(strcat("aP",num2str(ep)));
  endif
  
  if ( exist(strcat("aaP",num2str(ep))) == 0)
    Z=[];
  elseif(stat(strcat("aaP",num2str(ep))).size) == 0
    Z=[];
  else
    Z=load(strcat("aaP",num2str(ep)));
  endif
  figure
  try
  plot(X(:,1), X(:,2));
  catch
  end_try_catch
  xlabel('s'); 
  ylabel('a');
  axis([-1 1 -1 1]);
  title(strcat("P.",num2str(ep)));
  hold on;
  for i=1:size(Y,1)
    if( Y(i,3) > 0 )
      try
      plot(Y(i,1), Y(i,2), 'go', 'linewidth', 2);  
      catch
      end_try_catch
    elseif ( Y(i,3) < 0 )
      try
      plot(Y(i,1), Y(i,2), 'ro', 'linewidth', 2);
      catch
      end_try_catch
    else 
      try
      plot(Y(i,1), Y(i,2), 'bo');
      catch
      end_try_catch
    endif
  endfor
  
  for i=1:size(Z,1)
    try
    plot(Z(i,1), Z(i,2), '.', 'linewidth', 2); 
    catch
    end_try_catch
  endfor
  
endfunction

function load_pcs(ep) 
  if ( exist(strcat("pcs",num2str(ep))) == 0)
    X=[];
  elseif(stat(strcat("pcs",num2str(ep))).size) == 0
    X=[];
  else
    X=load(strcat("pcs",num2str(ep)));
  endif
  figure
  plot(X(:,1), X(:,2));
  xlabel('s'); 
  ylabel('ps');
  xlim([-1 1]);
  title(strcat("pcs.",num2str(ep))); 
endfunction

LIM=length(glob('P.[0-9]*'))-1;
START=max(LIM-15, 0);

%START=50
%
%LIM=65

%START=0
%LIM=2

for i=START:min(length(glob('P.[0-9]*'))-1, LIM)
  load_P(i)
%%  load_pcs(i)
endfor

fclose ("all")

%input("enter");