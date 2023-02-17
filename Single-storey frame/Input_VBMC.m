function  theta = Input_VBMC(t,y)
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
% y = measured input vector;
% 
% Output:
% theta = optimal hyperparameters vector
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

n = 2;              % Number of hyperparameters                  

% Specify prior distrubutions
prior_mu = [5e-3 1e-10];
prior_var = [5e-6 0.9e-20];

lpriorfun = @(x) ...
    -0.5*sum((x-prior_mu).^2./prior_var,2) ...
    -0.5*log(prod(2*pi*prior_var));

% Log joint distribution (unnormalized log posterior density)
fun = @(x)(log_likelihood(t,y,x) + lpriorfun(x));

% Bounds
LB = [1e-7 1e-13];                         % Lower bounds
UB = [1 1e-8];                             % Upper bounds
PLB = prior_mu - sqrt(prior_var);          % Plausible lower bounds
PUB = prior_mu + sqrt(prior_var);          % Plausible upper bounds

x0 = PLB + (PUB-PLB).*rand(1,n);    % Random point inside plausible box

% Calling VBMC function
options = vbmc('defaults');
% options.Plot = true;              % Uncomment to see real time plots

[vp,~,~] = vbmc(fun,x0,LB,UB,PLB,PUB,options);

Xs = vbmc_rnd(vp,3e5);              % Generate samples from the posterior

theta = zeros(1,n);
for i = 1:n
    theta(i) = mean(Xs(:,i));       % Set hyperparameters to posterior mean
end
end

function lml = log_likelihood(t,y,theta)
[~, lml] = Input_KF(t,y,theta,false);
end