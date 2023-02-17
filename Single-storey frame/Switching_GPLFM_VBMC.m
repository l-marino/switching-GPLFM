function [theta, elbo] = Switching_GPLFM_VBMC(t,y,u,m,c,k,p,S,I,J)
%--------------------------------------------------------------------------
% This function manages the settings regarding the inference of Gaussian 
% process hyperparameters, including priors, as well as lower and upper bounds 
% for the posterior distribution. The posterior of the hyperparameters is 
% inferred by the VBMC (Variational Bayes Monte Carlo) function developed by 
% Luigi Acerbi. Please check https://github.com/acerbilab/vbmc for
% instructions and further options on the use of VBMC.
% 
% Variables:
% t = observation time;
% y = observation vector;
% u = known driving force;
% m,c,k = mass, viscous damping and stiffness of the system;
% p = grade of the Matern kernel function;
% S = number of latent force models;
% I = number of Gaussians in ADF;
% J = number of Gaussians in EC;
% 
% Output:
% theta = optimal hyperparameters vector
% elbo = variational expected lower bound on the log marginal likelihood
%--------------------------------------------------------------------------
% To be used under the terms of the GNU General Public License 3.0
% (https://www.gnu.org/licenses/gpl-3.0.html).
%
% Author (copyright): Luca Marino, 2023
% e-mail: l.marino-1@tudelft.nl
% Version: 1.0.0
% Release date: Feb 17, 2023
% Code repository: https://github.com/l-marino/switching-GPLFM/
%--------------------------------------------------------------------------

rng('shuffle')   

n = 3;          % Number of hyperparameters                  

% Specify prior distrubutions
prior_mu = 20*ones(1,n);
prior_var = 100*ones(1,n);

lpriorfun = @(x)(-0.5*sum((x-prior_mu).^2./prior_var,2) ...
    -0.5*log(prod(2*pi*prior_var)));

% Log joint distribution (unnormalized log posterior density)
fun = @(x)(log_likelihood(t,y,u,m,c,k,p,S,I,J,x) + lpriorfun(x));

% Bounds
LB = [.1*ones(1,n-1) 10];           % Lower bounds
UB = 3e+2*ones(1,n);                % Upper bounds
PLB = prior_mu - sqrt(prior_var);          % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);          % Plausible upper bounds

x0 = PLB + (PUB-PLB).*rand(1,n);    % Random point inside plausible box

% Calling VBMC function
options = vbmc('defaults');
% options.Plot = true;              % Uncomment to see real time plots

[vp,elbo,~] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

Xs = vbmc_rnd(vp,3e5);              % Generate samples from the posterior

theta = zeros(1,n);
for i = 1:n
    theta(i) = mean(Xs(:,i));       % Set hyperparameters to posterior mean
end
end


function lml = log_likelihood(t,y,u,m,c,k,p,S,I,J,theta)
    [~, ~, ~, lml] = Switching_GPLFM_ADF_EC(t,y,u,theta,m,c,k,p,S,I,J,false);
end