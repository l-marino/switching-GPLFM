clear; clc; close all;
%--------------------------------------------------------------------------
% Application of the switching Gaussian process latent force model (GPLFM)
% to the identification of a base-excited single-storey frame with a
% friction contact between the top plate and a ground-fixed wall.
%
% This main script includes the uploading of the experimental data, the
% switching GPLFM setting, the parameter estimation and the nonlinear force 
% characterisation procedures.
%
% NOTE: the mass cannot be estimated and must be selected by the user.
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

%% Experimental data
% Data selection
weight = 550;               % Disc weight (g)
file_name= ['s_b_', num2str(weight),'g'];
load([file_name, '_','x','.mat'])
load([file_name, '_','y','.mat'])

t_obs = t - t(1); 
T = length(t_obs); tf = t_obs(end);
z_obs = 1e-3*x(1:length(t));            % (m)
u_obs_noisy = 1e-3*y(1:length(t));      % (m)

% Denoised base displacement and velocity estimation
theta_input = Input_VBMC(t_obs,u_obs_noisy);
input_states = Input_KF(t_obs,u_obs_noisy,theta_input,true);
u_obs = input_states(1,:);
du_obs = input_states(2,:);

rng('default') 

%% Switching GPLFM - inference
% SLDS settings
p = 0;                  % Matern kernel roughness (ni = p + 0.5)
S = 3;                  % Number of latent force models
I = 3;                  % Number of Gaussians for ADF
J = 3;                  % Number of Gaussians for EC

H = zeros(1,p+1); H(1) = 1;         % Latent force observation vector

% Initial guess system parameters
M_est = 3.0799;         % Mass (kg)
C_est = 0.6690;         % Viscous damping coefficient (Ns/m)
K_est = 1191;           % Stiffness (N/m)

% Inference hyperparameters (VBMC)
[theta, ~] = Switching_GPLFM_VBMC(t_obs,z_obs(:,1),[u_obs;du_obs],M_est,C_est,K_est,p,S,I,J);

% Inference latent states and force
[mu_smooth, Sigma_smooth, weights, lml] =...
    Switching_GPLFM_ADF_EC(t_obs,z_obs(:,1),[u_obs;du_obs],theta,M_est,C_est,K_est,p,S,I,J,true);

% Latent force models sequence
lfm_active = zeros(1,T);
for i=1:T
    [~, ind] = max(weights(:,i));
    lfm_active(i)  = ind;
end

% Assign latent states and force
displacement = mu_smooth(1,:);
velocity = mu_smooth(2,:);
friction_force = H*mu_smooth(3:end,:);

% %% Parameter estimation (optional)
% % Exploiting antisymmetry
% tol = 1e-2;         % Select a value ~=0 to neglect estimates around V=0
% ff = friction_force(lfm_active == 1 & abs(velocity) > tol);
% vv = velocity(lfm_active == 1 & abs(velocity) > tol);
% dd = displacement(lfm_active == 1 & abs(velocity) > tol);
% uu = u_obs(lfm_active == 1 & abs(velocity) > tol);
% duu = du_obs(lfm_active == 1 & abs(velocity) > tol);
% 
% dd(ff<0) = -dd(ff<0); vv(ff<0) = -vv(ff<0); uu(ff<0) = -uu(ff<0); 
% duu(ff<0) = -duu(ff<0); ff(ff<0) = -ff(ff<0);
% 
% if any(vv<0); dd = dd(vv>=0); uu = uu(vv>=0); duu = duu(vv>=0); 
%     ff = ff(vv>=0); vv = vv(vv>=0); end
% 
% % Determining parameters errors (least square method)
% model = @(F0,A1,A2) F0 + A1*(dd-uu) + A2*(vv-duu);
% residue = @(coeff) sum((model(coeff(1),coeff(2),coeff(3)) - ff).^2);
% coeff0 = [1 1 0];
% [coeff, ~] =  fmincon(residue,coeff0,[],[],[],[],[0 -K_est -C_est], [Inf Inf Inf]);
% disp(coeff)     % Select different initial values in case local minimum occurs
% coeff = num2cell(coeff);
% [F0,A1,A2] = deal(coeff{:});
% 
% Delta_K = A1;
% Delta_C = A2;
% 
% % Correct friction force (suggested if parameter error is small)
% if abs(Delta_K/K_est) > 1e-2
%     K_est_old = K_est;
%     K_est = K_est + Delta_K;
%     friction_force = friction_force - A1*displacement;     
% end
% if abs(Delta_C/C_est) > 1e-2
%     C_est_old = C_est;
%     C_est = C_est + Delta_C; 
%     friction_force = friction_force - A2*velocity;
% end
% 
% % Re-evaluate latent force with correct parameters (suggest for larger parameter errors)
% if abs(Delta_K/K_est) > 5e-2 || abs(Delta_C/C_est) > 5e-2
%     % Inference hyperparameters (VBMC)
%     theta_old = theta;
%     [theta, ~] = Switching_GPLFM_VBMC(t_obs,z_obs(:,1),[u_obs;du_obs],M_est,C_est,K_est,p,S,I,J);
% 
%     % Inference latent states and force
%     [mu_smooth, Sigma_smooth, weights, lml] =...
%         Switching_GPLFM_ADF_EC(t_obs,z_obs(:,1),[u_obs;du_obs],theta,M_est,C_est,K_est,p,S,I,J,true);
% 
%     % Latent force models sequence
%     lfm_active = zeros(1,T);
%     for i=1:T
%         [~, ind] = max(weights(:,i));
%         lfm_active(i)  = ind;
%     end
% 
%     % Assign latent states and force
%     displacement = mu_smooth(1,:);
%     velocity = mu_smooth(2,:);
%     friction_force = H*mu_smooth(3:end,:);
% end

acceleration = (-K_est*displacement - C_est*velocity - friction_force + u_obs)/M_est;

Ca = [-K_est/M_est -C_est/M_est -1/M_est*H]; Da = 1/M_est;   % Observation matrices for acceleration
st_z = zeros(2,T); st_f = zeros(1,T); st_a = zeros(1,T);

for t = 1:T
    st_z(:,t) = sqrt(diag(Sigma_smooth(1:2,1:2,t)));
    st_f(t) = sqrt(Sigma_smooth(3,3,t));
    st_a(t) = sqrt(Ca*Sigma_smooth(:,:,t)*Ca');
end

% Modal parameters
w0_est = sqrt(K_est/M_est);                 % Natural angular frequency (rad/s)
f0_est =(2*pi)\w0_est;                      % Natural frequency (Hz)
zeta_est = C_est/(2*sqrt(K_est*M_est));     % Damping ratio


%% Estimation static friction force
V = 3e-3;
eps_v = 1e-6;

if S > 1
    lfm_active_p = [lfm_active(2:end) NaN];
    if exist('F0','var') == 0; F0 = mean(abs(friction_force(lfm_active == 1))); end
    static_fr_values = abs(friction_force(abs(friction_force)>F0 & lfm_active == 2 & lfm_active_p ~= 2));
    static_fr_err = st_f(abs(friction_force)>F0 & lfm_active == 2 & lfm_active_p ~= 2);
    [static_fr_values_corr,i_corr] = rmoutliers(static_fr_values,"median");
    static_fr_err_corr = static_fr_err(~i_corr);
    st_mean = mean(static_fr_values_corr);
    st_st = std(static_fr_values_corr);
    if isnan(st_mean); st_mean = F0; end

    fo = fitoptions('Method','NonlinearLeastSquares',...
        'Lower',[.8*F0,0,0],...
        'Upper',[1.2*F0,0.1,0.1],...
        'StartPoint',[F0,0.04,0.02]);
    ft = fittype(@(F0,a,c,x) F0 + a*log((abs(x) + eps_v)/V) ...
        + (a + (st_mean - F0)/log(V/eps_v))*log(c + V./(abs(x) + eps_v)),'options',fo);
    [v_sliding, f_sliding] = ReorderSliding(velocity,friction_force,1e-2);
    f1 = fit(v_sliding',f_sliding',ft);   
    f1_b = f1.a + (st_mean - f1.F0)/log(V/eps_v);
else
    fo = fitoptions('Method','NonlinearLeastSquares',...
        'Lower',[.8*F0,0,0,0.01],...
        'Upper',[1.2*F0,0.1,0.2,0.1],...
        'StartPoint',[F0,0.04,0.065,0.02]);
    ft = fittype(@(F0,a,b,c,x) F0 + a*log((abs(x) + eps_v)/V) + b*log(c + V./(abs(x)+1e-6)),'options',fo);
    [v_sliding, f_sliding] = ReorderSliding(velocity,friction_force,1e-2);
    f1 = fit(v_sliding',f_sliding',ft);
    f1_b = f1.b;
end

v_fitted = logspace(-6,log10(abs(max(velocity))),1e+3);
rate_fitted = @(u)(f1.F0 + f1.a*log((abs(u) + eps_v)/V) + f1_b*log(f1.c + V./(abs(u)+eps_v)));
residuals = f_sliding - rate_fitted(v_sliding);
f_std = std(residuals)*ones(1,length(v_fitted));
f_std(1) = st_st;

friction_fitted = rate_fitted(v_fitted);

%% Evaluating performances
% Observed state identification errors (%)
NMSE_z = 100/(T*var(z_obs(:,1)))*sum((z_obs(:,1) - displacement').^2);

% Prediction error (%)
load_fun = griddedInterpolant(t_obs,K_est*u_obs+C_est*du_obs);
init_cond = [z_obs(1) (z_obs(2)-z_obs(1))/(t_obs(2)-t_obs(1))];
[~,z_check,~] = FrictionSDOF(M_est,C_est,K_est,rate_fitted,load_fun,tf,T,init_cond,0);
prediction_error = 100/(T*var(z_obs(:,1)))*sum((z_obs(:,1)-z_check(:,1)).^2);

%% Plots
h1 = figure(1);
subplot(5,1,1) 
plot(NaN,'-.r','linewidth',1);
hold on; plot(t_obs,displacement,'-b','linewidth',1)
hold on; patch([t_obs; flipud(t_obs)],[displacement - 3*st_z(1,:)  fliplr(displacement + 3*st_z(1,:))]',...
    'b','FaceAlpha',0.1,'linestyle','none');
hold on; plot(t_obs,z_obs,'-.r','linewidth',1,'handlevisibility','off')
ylabel('Displ. (m)'); xticklabels('');
legend('Measured', 'Estimated (mean)', 'Estimated (\pm 3\sigma intervals)',...
    'Orientation','horizontal','Position',[0.3,0.936,0.431,0.031])
set(gca,'linewidth',1.5,'FontSize',14)
subplot(5,1,2)
plot(t_obs,velocity,'-b','linewidth',1)
hold on; patch([t_obs; flipud(t_obs)],[velocity-3*st_z(2,:)  fliplr(velocity + 3*st_z(2,:))]',...
    'b','FaceAlpha',0.1,'linestyle','none');
ylabel('Vel. (m/s)'); xticklabels('');
ylim([-0.5 0.5])
set(gca,'linewidth',1.5,'FontSize',14)
subplot(5,1,3)
plot(t_obs, acceleration,'-b','linewidth',1)
hold on; patch([t_obs; flipud(t_obs)],[acceleration - 3*st_a  fliplr(acceleration+3*st_a)]',...
    'b','FaceAlpha',0.1,'linestyle','none');
ylabel('Acc. (m/s^2)'); xticklabels('');
ylim([-7 7])
set(gca,'linewidth',1.5,'FontSize',14)
subplot(5,1,4)
plot(t_obs,friction_force,'-b','linewidth',1)
hold on; patch([t_obs; flipud(t_obs)],[friction_force-3*st_f  fliplr(friction_force+3*st_f)]',...
    'b','FaceAlpha',0.2,'linestyle','none');
ylabel('Fric. force (N)'); xticklabels('');
set(gca,'linewidth',1.5,'FontSize',14)
ylim([-2 2])
subplot(5,1,5)
plot(t_obs,lfm_active,'-b','linewidth',1.2)
xlabel('Time (s)'); ylabel('Model')
set(gca,'linewidth',1.5,'FontSize',14)
ylim([0.8 3.2])
set(h1, 'Position', [1,1,1200,800]);
set(h1,'Units','Inches');
pos = get(h1,'Position');
set(h1,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

h2 = figure(2);
hm = heatmap(flipud(1-weights),'GridVisible','off','ColorbarVisible','off','colormap',gray);
xlabel('Time (s)'); ylabel('Model')
XLabels = (1:T)*(t_obs(2)-t_obs(1));
CustomXLabels = string(XLabels);
CustomXLabels(mod(XLabels,0.5) ~= 0) = " ";
CustomXLabels(1) = "0";
hm.XDisplayLabels = CustomXLabels;
s = struct(hm);
s.XAxis.TickLabelRotation = 0;
Ax = gca;
set(Ax,'FontSize',14)
set(h2, 'Position', [1,1,1200,600]);
set(h2,'Units','Inches');
pos = get(h2,'Position');
set(h2,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

h3 = figure(3);
hold on; plot(velocity,friction_force,'.b','linewidth',2,'markersize',4)
xlabel('Velocity (m/s)'); ylabel('Friction force (N)');
hold on; plot(NaN,'-g','linewidth',1.4);
hold on; patch([v_fitted fliplr(v_fitted)],...
    [rate_fitted(v_fitted)-1.96*f_std fliplr(rate_fitted(v_fitted)+1.96*f_std)],...
    'g','FaceAlpha',0.05,'linestyle','none');
hold on; patch([-v_fitted fliplr(-v_fitted)],...
    [-rate_fitted(-v_fitted)-1.96*f_std fliplr(-rate_fitted(-v_fitted)+1.96*f_std)],...
    'g','FaceAlpha',0.05,'linestyle','none','handlevisibility','off');
hold on; plot([-fliplr(v_fitted) v_fitted] ,...
    [-rate_fitted(-fliplr(v_fitted)) rate_fitted(v_fitted)],'-g','linewidth',1.4,'handlevisibility','off');
legend('Estimated','Fitted (mean)','Fitted (\pm 3\sigma intervals)',...
    'location','southeast','FontSize', 16, 'handlevisibility','off')
set(gca,'linewidth',1.5,'FontSize',20)
set(h3, 'Position', [1,1,800,600]);
set(h3,'Units','Inches');
pos = get(h3,'Position');
set(h3,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

h4 = figure(4);
plot(NaN,'.r','linewidth',2,'markersize',6)
hold on; plot(displacement,friction_force,'.b','linewidth',2)
xlabel('Displacement (m)'); ylabel('Friction force (N)');
legend('Estimated','orientation','horizontal')
set(gca,'linewidth',1.5,'FontSize',20)
set(h4, 'Position', [1,1,800,600]);
set(h4,'Units','Inches');
pos = get(h4,'Position');
set(h4,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])

h5 = figure(5);
t_corr = t_obs(abs(friction_force)>F0 & lfm_active == 2 & lfm_active_p ~= 2); t_corr = t_corr(~i_corr);
errorbar(t_corr, static_fr_values_corr, 1.96*static_fr_err_corr,'ob','linewidth',1.3)
hold on; plot(NaN,'-g','linewidth',1.4);
hold on; patch([0 t_obs(end) t_obs(end) 0], [st_mean-1.96*st_st st_mean-1.96*st_st st_mean+1.96*st_st st_mean+1.96*st_st],...
    'g','FaceAlpha',0.05,'linestyle','none');
hold on; plot([0 t_obs(end)], [st_mean, st_mean],'-g','linewidth',1.4,'handlevisibility','off');
xlabel('Time (s)'); ylabel('Static friction force (N)');
legend('Estimated','Fitted (mean)','Fitted (\pm 3\sigma intervals)',...
    'location','south','NumColumns', 2, 'FontSize', 16)
set(gca,'linewidth',1.5,'FontSize',20)
set(h5, 'Position', [1,1,800,600]);
set(h5,'Units','Inches');
pos = get(h5,'Position');
set(h5,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
