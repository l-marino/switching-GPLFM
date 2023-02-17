function [t_obs,y_obs,F_obs,t,x,F] = FrictionSDOF(M,C,K,rate_fun,load_fun,tf,N_obs,x0,noise)
%--------------------------------------------------------------------------
% This function simulates the response of a single-degree-of-freedom system
% with a friction contact between mass and a fixed wall. The excitation and
% the friction rate-dependent law can be assigned by the user as functions.
% The integration approach is the event-driven Runge-Kutta (4,5) algorithm
% from https://www.sciencedirect.com/science/article/pii/S0022460X22002966.
% 
% Variables:
% M,C,K = mass, viscous damping and stiffness of the system;
% rate_fun = user defined rate-dependent friction law
% load_fun = user defined forcing function
% tf = simulation final time;
% T = number of samples (for the results interpolation on a fixed time step 
%     vector);
% x0 = initial position and velocity
% noise = signal-to-noise ratio (dB) of the white noise to be added to the
%         simulated displacement (leave "0" for noiseless signal)
% 
% Output: 
% t_obs,y_obs,F_obs = fixed-step time, states and friction force 
%                     vectors - measured states will include noise;
% t,x,F = time, states and friction force from time integration.
%
% NOTE: for base excitation, the load function must be formulated as the 
% equivalent mass excitation K*u + C*u', where u and u' are the base
% displacement and velocity.
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

% Time duration choice
t0 = 0;             % Initial time
t = t0; x = x0;     % Initialising time and displ. vectors

% Observation time vector
t_obs = linspace(t0,tf,N_obs);
dt_obs = t_obs(2) - t_obs(1);

% Initial friction force value
if x0(2) == 0
    F = min(abs(load_fun(t0) - K*x0(1)), rate_fun(0))*...
            sign(load_fun(t0) - K*x0(1));
else
    F = rate_fun(x0(2))*sign(x0(2));
end

% ODE settings
Ev1 = @(t,x)myEvent(t,x);
options = odeset('AbsTol',1e-5,'RelTol',1e-8,'Events',Ev1);


% ODE function
FricDamp = @(t,x) [x(2,1); M\(load_fun(t)-C*x(2,1)-K*x(1,1)-rate_fun(x(2,1))*sign(x(2,1)))];

% Main loop
while t0 < tf
    if abs(load_fun(t0)-C*x0(2)-K*x0(1))>=abs(rate_fun(0))      % sliding phase
        [t_new, x_new] = ode23s(FricDamp, [t0 tf], x0, options);
        t_new = t_new(2:end);
        x_new = x_new(2:end,:);
        F_new = rate_fun(x_new(:,2)).*sign(x_new(:,2));
    else        % sticking phase
        % find the point where sticking conditions are no longer verified
        F(end) = load_fun(t(end))-K*x(end,1);
        myfun = @(c1,c2,x_stop,k)(abs(load_fun(c1)-k*x_stop)-c2);
        x_stop = x0(1); c2 = rate_fun(0);
        fun = @(c1)myfun(c1,c2,x_stop,K);
        incd = 1e-2;
        while sign(fun(t0))*sign(fun(t0+incd)) ~= -1 && t0<=tf 
            t0 = t0+incd;
        end
        if t0 < tf
            tzero = fzero(fun,[t0 t0+incd]);
        else
            tzero = tf;
        end
        t_new = [(t(end)+dt_obs:dt_obs:tzero)'; tzero + 1e-6];      % This is the point
        x_new = [x_stop*ones(length(t_new),1) zeros(length(t_new),1)];
        F_new = load_fun(t_new)-K*x_stop;
    end
    
    t = [t; t_new];
    x = [x; x_new];
    F = [F; F_new];
    t0 = t(end);
    x0 = x(end,:);    
end

y_obs = interp1(t,x,t_obs);                    % clean displacement
if noise ~= 0
    y_obs(:,1) = awgn(y_obs(:,1),noise,'measured');      % noisy displacement
end

t_obs = t_obs';
F_obs = interp1(t,F,t_obs);
end


%% FUNCTION MYEVENT1
function [velocity,isterminal,direction]= myEvent(~,x)
velocity = x(2);
isterminal = 1;
direction = 0;
end