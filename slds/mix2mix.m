function [newcoeff, newmean, newcov] = mix2mix(coeff, mean, cov, I)

%
% D.Barber : Expectation Correction for smoothed Inference in Switching Linear
% Dynamical Systems (Journal of Machine Learning Research 2007)
%
% Code : Bertrand Mesot 15 Nov 2006
  
  
% fit a mixture of Gaussians with another mixture of Gaussians
% (but with a smaller number of components I), by retaining the
% I-1 most probable coeffs, and merging the rest    

newcoeff = coeff;
newmean  = mean;
newcov   = cov;

L = length(newcoeff);

if L > I
  [~,ind] = sort(coeff);

  tomerge   = ind(1:L-I+1);
%   notmerged = setdiff(1:L,tomerge);
  
  sump = sum(coeff(tomerge));
  
  if sump ~= 0
    condp = coeff(tomerge) ./ sump; 
  else
    n     = length(tomerge);
    condp = ones(1,n) / n;
  end
    
  [mergedmean, mergedcov] = ...
      matmixtosingle(condp, mean(:,tomerge), cov(:,:,tomerge));
  
 % for i=1:size(mergedcov,3)
 %   if det(mergedcov(:,:,i))<0
 %     disp('break in merge routine')
 %     keyboard
 %   end
 % end
  
  newcoeff(tomerge)   = [];
  newmean(:,tomerge)  = [];
  newcov(:,:,tomerge) = [];
  
  newcoeff(I)   = sump;
  newmean(:,I)  = mergedmean;
  newcov(:,:,I) = mergedcov;
end
  
function [m, S] = matmixtosingle(coeff, means, cov)

% fit a mixture of Gaussians with a single Gaussian, so that the
% first and second moments of the fitted Gaussian match the
% mixture first and second moments.

n=size(means,2);

m=zeros(size(means,1),1);
S=zeros(size(means,1));

for i=1:n
    m=m+coeff(i)*means(:,i);
    S=S+coeff(i)*(cov(:,:,i)+means(:,i)*means(:,i)');
end
S=S-m*m';

  
  

  
  