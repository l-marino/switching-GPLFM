function [s] = mylogsum(x)

%
% D.Barber : Expectation Correction for smoothed Inference in Switching Linear
% Dynamical Systems (Journal of Machine Learning Research 2007)
%
% Code : Bertrand Mesot 15 Nov 2006
  
  
s = x(1);

for k=2:length(x)
  s = logadd(s, x(k));
end

function [s] = logadd(x1, x2)

if x1 == x2
  s = x1 + log(2);
else
  if x1 > x2
    s = x1 + log(1+exp(x2-x1));
  else
    s = x2 + log(1+exp(x1-x2));
  end
end