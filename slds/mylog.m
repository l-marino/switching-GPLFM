function [logx] = mylog(x)

%
% D.Barber : Expectation Correction for smoothed Inference in Switching Linear
% Dynamical Systems (Journal of Machine Learning Research 2007)
%
% Code : Bertrand Mesot 15 Nov 2006  
  
[nr, nc] = size(x);

logx = -inf*ones(nr,nc);

for c=1:nc
  for r=1:nr
    if x(r,c) ~= 0
      logx(r,c) = log(x(r,c));
    end  
  end
end