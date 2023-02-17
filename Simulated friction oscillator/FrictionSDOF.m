function [t_obs,y_obs,F_obs,t,x,F] = FrictionSDOF(M,C,K,rate_fun,load_fun,tf,N_obs,x0,noise)

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