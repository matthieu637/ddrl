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
  zlim([-30 30]);
  title(strcat("Q.",num2str(ep)));
endfunction

function load_V(ep) 
  X=load(strcat("Q.",num2str(ep)));
  
  if ( exist(strcat("aQ.",num2str(ep))) == 0)
    Y = [];
  elseif ((stat(strcat("aQ.",num2str(ep))).size) == 0)
    Y=[];
  else
    Y=load(strcat("aQ.",num2str(ep)));
  endif
  
  figure
  plot(X(:,1), X(:,2)); hold on;
  xlabel('s');
  ylabel('a');
  axis([-1 1 -10 10])
  
  if( size(Y,1) != 0 )
    plot(Y(:,1), Y(:,2), 'r*', 'linewidth', 6);
  endif

  title(strcat("V.",num2str(ep)));
endfunction

function fig=load_Vi(ep, st) 
  X=load(strcat("V.",num2str(ep), '.' , num2str(st) ));
  
  if ( exist(strcat("aQ.",num2str(ep))) == 0)
    Y = [];
  elseif ((stat(strcat("aQ.",num2str(ep))).size) == 0)
    Y=[];
  else
    Y=load(strcat("aQ.",num2str(ep)));
  endif
  
  fig=figure;
  plot(X(:,1), X(:,2)); hold on;
  xlabel('s');
  ylabel('a');
  axis([-1 1 -10 10])
  
  if( size(Y,1) != 0 )
    plot(Y(:,1), Y(:,2), 'r*', 'linewidth', 6);
  endif

  title(strcat("V.",num2str(ep)));
endfunction


LIM=length(glob('P.[0-9]*'))-1;
START=max(LIM-15, 0);

%START=50
%
%LIM=100

%START=0
%LIM=3

%for i=START:min(length(glob('V.[0-9]*'))-1, LIM)
%  globed = glob(strcat('aaQ.', num2str(i), '.[0-9]*'));
%  substart = str2num(strsplit(globed{1},'.')(end){1});
%  for j=substart:substart+length(globed)-1
%    fig = load_Vi(i, j);
%    
%    X=load(strcat('aaQ.',num2str(i),'.',num2str(j)));
%    
%    plot(X(:,1), X(:,2), '.');
%    title(strcat('aaQ.',num2str(i),'.',num2str(j)));
%    
%%    print(fig, strcat('out/aaQ.',num2str(i),'.',num2str(j),'.png'));
%%    close(fig);  
%  endfor
%endfor

for i=START:min(length(glob('Q.[0-9]*'))-1, LIM)
  load_V(i)
  
  X=load(strcat('aaQ.',num2str(i),'.',num2str(0)));
  plot(X(:,1), X(:,2), '.');
endfor

%for i=START:min(length(glob('Q.[0-9]*'))-1, LIM)
%  load_Q(i)
%endfor

fclose ("all")
%input("enter");