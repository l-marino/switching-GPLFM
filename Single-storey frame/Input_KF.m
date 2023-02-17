function  [mu_smooth, lml] = Input_KF(t,y,theta,prediction)
%--------------------------------------------------------------------------
% This function computes the posterior distribution of the latent states for
% given measured displacements. The posteriors are computed via Kalman filtering 
% and RTS smoothing.
% 
% Variables:
% t = observation time;
% y = observation vector;
% theta = optimal hyperparameters;
% prediction = smoothing step required (1 = yes, 0 = no)
% 
% Output:
% mu_smooth = mean smoothed posterior;
% lml = log marginal likelihood;
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

%% Initialisation
T = length(t);
delta_t = t(2)- t(1);
X = 2;

% State-space matrices
A = [1 delta_t; 0 1];
Q = [.5*delta_t^2; delta_t]*[.5*delta_t^2; delta_t]'*theta(1);
C = [1 0];
R = theta(2);

% Allocate memory
mu_pred = zeros(X,T);
Sigma_pred = zeros(X,X,T);
mu_upd = zeros(X,T);
Sigma_upd = zeros(X,X,T);
lml_t = zeros(1,T);

%% Kalman filtering
for t = 1:T
    % Prediction density
    if t == 1
        mu_pred(:,t) = [y(1) (y(2)-y(1))/delta_t];
        Sigma_pred(:,:,t) = 0.1*eye(X);
    else
        mu_pred(:,t) = A*mu_upd(:,t-1);
        Sigma_pred(:,:,t) = A*Sigma_upd(:,:,t-1)*A' + Q;
    end

    r = y(t) - C*mu_pred(:,t);
    G = C*Sigma_pred(:,:,t)*C' + R;

    K   = (C*Sigma_pred(:,:,t))'/G;
    IKC = eye(X) - K*C;

    % Filtered density
    mu_upd(:,t) = mu_pred(:,t) + K*r;
    Sigma_upd(:,:,t) = IKC*Sigma_pred(:,:,t)*IKC' + K*R*K';
    
    % Log likelihood
    lml_t(t) = - 0.5*r'/G*r - 0.5*log(2*pi) - 0.5*trace(logm(G));
end

lml = mylogsum(lml_t(t));

%% Rauch-Tung-Striebel smoothing 
mu_smooth = mu_upd;
Sigma_smooth = Sigma_upd;

if prediction == 1
    for t = T-1:-1:1
        J = Sigma_upd(:,:,t)*A'/(Sigma_pred(:,:,t+1)+1e-16*eye(X));
        mu_smooth(:,t) = mu_upd(:,t)+J*(mu_smooth(:,t+1)-mu_pred(:,t+1));
        Sigma_smooth(:,:,t) = Sigma_upd(:,:,t)+J*(Sigma_smooth(:,:,t+1)-Sigma_pred(:,:,t+1))*J';
    end
end
end