clear all;
close all;

%debug_on_error(1)

function load_Q(ep) 
  X=load(strcat("Q.",num2str(ep)));
  figure
  [xx,yy] = meshgrid (linspace (-1,1,50));
  griddata(X(:,1), X(:,2), X(:,3), xx, yy, "linear"); xlabel('theta'); ylabel('a');
  xlabel('s');
  ylabel('a');
  zlim([-20 20]);
  title(strcat("Q.",num2str(ep)));
endfunction

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
  plot(X(:,1), X(:,2));
  xlabel('s'); 
  ylabel('a');
  axis([-1 1 -1 1]);
  title(strcat("P.",num2str(ep)));
  hold on;
  for i=1:size(Y,1)
    if( Y(i,3) > 0 )
      plot(Y(i,1), Y(i,2), 'go', 'linewidth', 2);  
    elseif ( Y(i,3) < 0 )
      plot(Y(i,1), Y(i,2), 'ro', 'linewidth', 2);
    else 
      plot(Y(i,1), Y(i,2), 'bo', 'linewidth', 2);
    endif
  endfor
  
  for i=1:size(Z,1)
    plot(Z(i,1), Z(i,2), '.', 'linewidth', 2);  
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

%for i=0:length(glob('Q.[0-9]*'))-1
%  load_Q(i)
%endfor

for i=0:length(glob('P.[0-9]*'))-1
  load_P(i)
%  load_pcs(i)
endfor

%input("enter");