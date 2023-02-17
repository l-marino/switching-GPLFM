function [mu, Sigma, weights, lml] = Switching_GPLFM_ADF_EC(t,y,u,theta,m,c,k,p,S,I,J,prediction)
%--------------------------------------------------------------------------
% This function computes the posterior distribution of the latent states 
% and (discontinuous) nonlinear force of a (mass-excited) mass-spring-damper 
% system with a  friction contact. The posteriors are computed via assumed 
% density filtering (ADF) and the expectation-correction (EC) algorithm from 
% Barber (2006).
% 
% Variables:
% t = observation time;
% y = observation vector;
% u = known driving force;
% theta = optimal hyperparameters;
% m,c,k = mass, viscous damping and stiffness of the system;
% p = grade of the Matern kernel function;
% S = number of latent force models;
% I = number of Gaussians in ADF;
% J = number of Gaussians in EC;
% prediction = smoothing step required (1 = yes, 0 = no)
% 
% Output:
% mu = mean smoothed posterior;
% Sigma = covariance of the smoothed posterior;
% weights = likelihoods of the latent force models;
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
% Assigning hyperparameters values 
sigma_f2 = theta(1);                    % Variance amplitude latent force
l = theta(2);                           % Length-scale latent force
sigma_n2 = 1e-12*theta(3);              % Variance measurement noise

delta_t = t(2) - t(1);                  % Time step
T = length(t);                          % Time vector size
Y = size(y,2);                          % Observation vector size
Xf = 1 + p;                             % Late
X = 2 + Xf;                             % Augmented state vector size

% Allocate memory
A = zeros(X,X,S);
B = zeros(X,1,S);
C = zeros(1,X); C(1) = 1;
Q = zeros(X,X,S);

pMean = zeros(X,S);
pCov = zeros(X,X,S);
for s=1:S
    pCov(:,:,s) = [5e-3 0 0; 0 0.1 0; 0 0 0.05];        % To be defined by the user
end

prior = zeros(S,1); prior(1) = 1;
stran = zeros(S,S);

logr = -inf*ones(S,T);
logfpmix = -inf*ones(I,S,T);
log_p_itm_stm_st_g_y1t = zeros(I,S,S);
mu_upd = zeros(X,I,S,T);
Sigma_upd = zeros(X,X,I,S,T);
mu_upd_s  = zeros(X,I,S,S);
Sigma_upd_s = zeros(X,X,I,S,S);
mu_s = zeros(X,S,T);
Sigma_s = zeros(X,X,S,T);
mu = zeros(X,T);
Sigma = zeros(X,X,T);

if prediction == true           % Only if EC is performed
    logl = -inf*ones(S,T);
    logbpmix = -inf*ones(J,S,T);
    log_p_it_st_g_jtp_stp_y1T_xtp  = -inf*ones(J,S,I,S);
    log_p_jtp_stp_it_g_st_y1T = -inf*ones(J,S,I,S);
    m_xt_y1T   = zeros(X,J,S,I,S);
    Cov_xt_y1T = zeros(X,X,J,S,I,S);
    mu_smooth = zeros(X,J,S,T);
    Sigma_smooth = zeros(X,X,J,S,T);
end

% Definining model transition parameters
rho = 0.92;         % To be defined by the user

if S == 1
    stran = 1;
else
    for s=1:S-1
        stran(s,S) = 1/(S-1);
        stran(S,s) = 1-rho;
        stran(s,s) = rho;
    end
end

logstran = mylog(stran);

%% Probabilistic model
% System state-space model matrices
A_cs = [0 1; -k/m -c/m];
B_cs = [0; 1/m];

% Latent force state-space model matrices
model.ss{1}.make_ss = @cf_matern_to_ss;
model.ss{1}.lengthScale = l;
model.ss{1}.magnSigma2 = sigma_f2;
model.ss{1}.nu = p + 0.5;
model.ss{1}.N = 1;
model.ss{1}.opt = {'magnSigma2','lengthScale'};

[~,pnames] = ss_pak(model);
[A_cf,L,Qc,H] = ss_stack(model, pnames);

% Sliding model (latent force model 1)
st = 1;

B_cf = [zeros(1,Xf); -1/m*H];
A_c = [A_cs B_cf; zeros(Xf,2) A_cf];
B_c = [B_cs; zeros(Xf,1)];

A(:,:,st) = expm(A_c*delta_t);
B(:,:,st) = A_c\(A(:,:,st)-eye(X))*B_c;

Q_c = zeros(X);
Q_c(3:end,3:end) = L*Qc*L';
P_ss = lyap(A_c,Q_c);
Q(:,:,st) = P_ss-A(:,:,st)*P_ss*A(:,:,st)';
Q(1,1,st) = 1e-16;

R = sigma_n2; 

if S > 2
    % Resetting model (latent force model 3)
    st = S;

    A(1:2,1:2,st) = expm(A_cs*delta_t);
    B(:,:,st) = A_c\(A(:,:,st)-eye(X))*B_c;
    Q(3:end,3:end,st) = squeeze(pCov(3,3,3));
    Q(1,1,st) = 1e-16;
end

%% Assumed density filter (ADF)
% Case t = 1
for st = 1:S
    if prior(st) == 0
        logr(st,1)       = -inf;
        logfpmix(:,st,1) = -inf;
        continue;
    end

    % p(x1, y1 | s1)
    mu_pred_t = pMean(:,st);                % Prediction mean states
    y_pred_t = C*mu_pred_t;                 % Prediction mean observations

    Sigma_pred_t = pCov(:,:,st);            % Prediction cov states
    G = C*Sigma_pred_t*C' + R;              % Prediction cov observations
    G = 0.5*(G + G');

    K   = (C*Sigma_pred_t)'/ G;           % K from Kalman filter
    IKC = eye(X) - K*C;                   % I-KC from Kalman filter
    r = y(1,:) - y_pred_t;                  % r from Kalman filter

    Sigma_upd_t = IKC*Sigma_pred_t*IKC' + K*R*K';
    Sigma_upd_t = 0.5*(Sigma_upd_t + Sigma_upd_t');   % update cov

    mu_upd(:,1,st,1) = mu_pred_t + K*r;       % Filtered mean states
    Sigma_upd(:,:,1,st,1) = Sigma_upd_t;        % Filtered cov states

    % p(s1 | y1) ~ p(y1 | s1) p(s1) (log likelihood of the switch variable)
    logr(st,1) = log(prior(st)) - 0.5*(r'/G*r + trace(logm(G)) + Y*log(2*pi));

    % p(i1 | s1) (log likelihood of the Gaussian component)
    logfpmix(1,st,1) = 0;     
end 

log_p_y1    = mylogsum(logr(:,1));
logr(:,1) = logr(:,1) - log_p_y1;

lml = log_p_y1;                              % update log likelihood

% Case t = 2:T
for t = 2:T
    for stm = 1:S
        for itm = 1:I
            for st = 1:S
                if logstran(st,stm) == -inf || ...
                        logr(stm,t-1) == -inf || ...
                        logfpmix(itm,stm,t-1) == -inf

                    log_p_itm_stm_st_g_y1t(itm,stm,st) = -inf;
                    continue;
                end

                % p(xt, yt | xtm, utm, st, stm, itm)
                if st ~= 2          % Excluding sticking model
                    mu_pred_t  = A(:,:,st)*mu_upd(:,itm,stm,t-1) + B(:,:,st)*u(t-1);

                    Sigma_pred_t = A(:,:,st)*Sigma_upd(:,:,itm,stm,t-1)*A(:,:,st)' + Q(:,:,st);
                    Sigma_pred_t = 0.5*(Sigma_pred_t + Sigma_pred_t');
                else
                    mu_pred_t  = [mu_upd(1,itm,stm,t-1); 0; u(t-1) - k*mu_upd(1,itm,stm,t-1)];

                    Sigma_pred_t = squeeze(Sigma_upd(:,:,itm,stm,t-1));
                    Sigma_pred_t = 0.5*(Sigma_pred_t + Sigma_pred_t');
                end

                y_pred_t  = C*mu_pred_t;
                G = C*Sigma_pred_t*C' + R;
                G = 0.5*(G + G');

                K   = (C*Sigma_pred_t)' / G;
                IKC = eye(X) - K*C;

                Sigma_upd_t = IKC*Sigma_pred_t*IKC' + K*R*K';
                Sigma_upd_t = 0.5*(Sigma_upd_t+Sigma_upd_t');

                r = y(t,:) - y_pred_t;

                Sigma_upd_s(:,:,itm,stm,st) = Sigma_upd_t;
                mu_upd_s(:,itm,stm,st) = mu_pred_t + K*r;

                % p(itm, stm, st | y1:t) ~ p(yt | st, stm, itm, y1:t)
                %                          * p(st | stm)
                %                          * p(itm | stm, y1:tm)
                %                          * p(stm | y1:tm)
                log_p_yt_g_st_stm_itm_y1tm = ...
                    - 0.5*(r'/G*r + 0.5*trace(logm(G)) + Y*log(2*pi));

                log_p_itm_stm_st_g_y1t(itm,stm,st) = ...
                    + log_p_yt_g_st_stm_itm_y1tm ...
                    + logstran(st,stm) ...
                    + logfpmix(itm,stm,t-1) ...
                    + logr(stm,t-1);
            end % end st loop
        end % end itm loop
    end % end stm loop

    log_p_yt_g_y1tm = -inf;

    for st=1:S
        for stm=1:S
            log_p_yt_g_y1tm = mylogsum([log_p_yt_g_y1tm mylogsum(log_p_itm_stm_st_g_y1t(:,stm,st))]);
        end
    end

    lml = lml + log_p_yt_g_y1tm;

    % p(st, stm, itm | y1:t)
    log_p_itm_stm_st_g_y1t = log_p_itm_stm_st_g_y1t - log_p_yt_g_y1tm;

    % Collapse to mixture of I Gaussians
    for st=1:S

        % p(st | y1:t)
        log_p_st_g_y1t = -inf;

        for stm=1:S
            log_p_st_g_y1t = ...
                mylogsum([log_p_st_g_y1t mylogsum(log_p_itm_stm_st_g_y1t(:,stm,st))]);
        end

        logr(st,t) = log_p_st_g_y1t;

        if log_p_st_g_y1t ~= -inf
            len = S*I;

            coeff = zeros(len,1);
            mean  = zeros(X,len);
            cov   = zeros(X,X,len);

            nmix = 0;

            for stm=1:S
                for itm=1:I
                    % p(stm, itm | y1:t, st)
                    p_itm_stm_g_st_y1t = exp(log_p_itm_stm_st_g_y1t(itm,stm,st) - log_p_st_g_y1t);

                    if p_itm_stm_g_st_y1t ~= 0
                        nmix = nmix + 1;

                        coeff(nmix)   = p_itm_stm_g_st_y1t;
                        mean(:,nmix)  = mu_upd_s(:,itm,stm,st);
                        cov(:,:,nmix) = Sigma_upd_s(:,:,itm,stm,st);
                    end
                end
            end

            [newcoeff, newmean, newcov] = ...
                mix2mix(coeff(1:nmix), mean(:,1:nmix), cov(:,:,1:nmix), I);

            L = length(newcoeff);

            logfpmix(1:L,st,t) = log(newcoeff);
            mu_upd(:,1:L,st,t) = newmean;
            Sigma_upd(:,:,1:L,st,t) = newcov;
        end % end if
    end % end st loop
end % end t loop

if prediction == false

    weights = exp(logr);            % Weights of the latent force models

    for t = 1:T
        for st=1:S
            % Filtered distribution for each latent force model 
            mu_s(:,st,t) = exp(logfpmix(:,st,t))'*squeeze(mu_upd(:,:,st,t))';
            for it = 1:J
                Sigma_s(:,:,st,t) = Sigma_s(:,:,st,t) + exp(logfpmix(it,st,t))*squeeze(Sigma_upd(:,:,it,st,t));
            end

            % Overall filtered distribution 
            Sigma(:,:,t) = Sigma(:,:,t) + weights(st,t)*squeeze(Sigma_s(:,:,st,t));
        end
        mu(:,t) = weights(:,t)'*squeeze(mu_s(:,:,t))';
    end

else

%% Expactation-correction algorithm (from Barber)
    % message initialisation
    if J < I
        % Collapse to a mixture of J Gaussians
        for st=1:S
            coeff = exp(logfpmix(:,st,T));
            mean = mu_upd(:,:,st,T);
            cov = Sigma_upd(:,:,:,st,T);

            [newcoeff, newmean, newcov] = mix2mix(coeff, mean, cov, J);

            logl(st,T) = logr(st,T);
            logbpmix(:,st,T) = log(newcoeff);
            mu_smooth(:,:,st,T) = newmean;
            Sigma_smooth(:,:,:,st,T) = newcov;
        end       
    elseif J == I
        logl(:,T) = logr(:,T);
        logbpmix(1:I,:,T) = logfpmix(:,:,T);
        mu_smooth(:,1:I,:,T) = mu_upd(:,:,:,T);
        Sigma_smooth(:,:,1:I,:,T) = Sigma_upd(:,:,:,:,T);
    else
        disp('J cannot be bigger than I');
        keyboard
    end

    % Time steps T-1:-1:1
    for t=T-1:-1:1
        for stp=1:S
            if stp ~=2
                for st=1:S
                    for it=1:I
                        if logstran(stp,st) == -inf || ...
                                logr(st,t) == -inf || ...
                                logfpmix(it,st,t) == -inf

                            for jtp=1:J
                                log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st) = -inf;
                            end
                        end

                        % p(xt, xtp | it, st, stp, y1:t)
                        mean = mu_upd(:,it,st,t);
                        cov  = Sigma_upd(:,:,it,st,t);

                        m_xtp_g_y1t = A(:,:,stp)*mean + B(:,:,stp)*u(t);

                        Cov_xtp_xt_g_y1t  = A(:,:,stp)*cov;
                        Cov_xt_xtp_g_y1t  = Cov_xtp_xt_g_y1t';
                        Cov_xtp_xtp_g_y1t = ...
                            A(:,:,stp)*cov*A(:,:,stp)' + Q(:,:,stp) + 1e-14*eye(X);

                        Cov_xtp_xtp_g_y1t = 0.5*(Cov_xtp_xtp_g_y1t + Cov_xtp_xtp_g_y1t');

                        leftA = Cov_xt_xtp_g_y1t/Cov_xtp_xtp_g_y1t;

                        % p(stp, it, st | y1:t)

                        log_p_stp_it_st_g_y1t = ...
                            logstran(stp,st) + logr(st,t) + logfpmix(it,st,t);

                        % ...

                        avtmp = - 0.5*trace(log(Cov_xtp_xtp_g_y1t));

                        % ...

                        inv_Cov_xtp_xtp_g_y1t = inv(Cov_xtp_xtp_g_y1t);

                        for jtp=1:J
                            % p(xt, xtp | it, st, jtp, stp, v1:T)
                            % Joseph's stabilized update form

                            z = mu_smooth(:,jtp,stp,t+1) - m_xtp_g_y1t;

                            m_xt_y1T(:,jtp,stp,it,st) = mean + leftA * z;

                            tmp = eye(X) - leftA * A(:,:,stp);

                            Xx = tmp*cov*tmp' + leftA ...
                                * (Q(:,:,stp) + Sigma_smooth(:,:,jtp,stp,t+1)) * leftA';

                            Xx = 0.5*(Xx+Xx');

                            Cov_xt_y1T(:,:,jtp,stp,it,st) = Xx;

                            % <p(it, st | xtp, stp, y1:t)>_p(xtp | jtp, stp, y1:T)

                            tmp = inv_Cov_xtp_xtp_g_y1t;

                            av_p_it_st_g_xtp_stp_y1t = avtmp - 0.5*sum((z'*tmp).*z', 2);

                            % p(it, st | jtp, stp, y1:t, xtp) ~
                            %       p(stp, it, st | y1:t) <p(it, st | xtp, stp, y1:t)>

                            log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st) = ...
                                log_p_stp_it_st_g_y1t + av_p_it_st_g_xtp_stp_y1t;

                        end % end jtp loop
                    end % end it loop
                end % end st loop
            else
                for st=1:S
                    for it=1:I
                        if logstran(stp,st) == -inf || ...
                                logr(st,t) == -inf || ...
                                logfpmix(it,st,t) == -inf

                            for jtp=1:J
                                log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st) = -inf;
                            end
                        end

                        % p(xt, xtp | it, st, stp, y1:t)
                        mean = mu_upd(:,it,st,t);
                        cov = Sigma_upd(:,:,it,st,t);

                        % p(stp, it, st | y1:t)
                        log_p_stp_it_st_g_y1t = ...
                            logstran(stp,st) + logr(st,t) + logfpmix(it,st,t);

                        for jtp = 1:J
                            m_xt_y1T(:,jtp,stp,it,st) = mean;
                            Cov_xt_y1T(:,:,jtp,stp,it,st) = leftA*(cov + Q(:,:,S))*leftA';
                            log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st) = log_p_stp_it_st_g_y1t;
                        end
                    end % end it loop
                end % end st loop
            end % end if
        end % end stp loop

        % p(it, st | stp, jtp, y1:t, xtp) =
        %          p(it, st, stp, jtp | y1:t, xtp) / p(stp, jtp | y1:t, xtp)
        for stp=1:S
            for jtp=1:J
                log_p_jtp_stp_g_y1T_xtp = -inf;

                for st=1:S
                    for it=1:I
                        tmp = log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st);
                        log_p_jtp_stp_g_y1T_xtp = mylogsum([log_p_jtp_stp_g_y1T_xtp tmp]);
                    end
                end

                if log_p_jtp_stp_g_y1T_xtp ~= -inf
                    log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,:,:) = ...
                        log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,:,:) - log_p_jtp_stp_g_y1T_xtp;
                end
            end
        end

        % p(jtp, stp, it | st, y1:T) = p(it, st | jtp, stp, y1:t, xtp)
        %                              * p(jtp | stp, y1:T) p(stp | y1:T)
        for st=1:S
            log_p_st_g_y1T = -inf;

            for it=1:I
                for stp=1:S
                    for jtp=1:J
                        tmp = ...
                            log_p_it_st_g_jtp_stp_y1T_xtp(jtp,stp,it,st) ...
                            + logbpmix(jtp,stp,t+1) ...
                            + logl(stp,t+1);

                        log_p_jtp_stp_it_g_st_y1T(jtp,stp,it,st) = tmp;

                        log_p_st_g_y1T = mylogsum([log_p_st_g_y1T tmp]);
                    end
                end
            end

            if log_p_st_g_y1T ~= -inf
                log_p_jtp_stp_it_g_st_y1T(:,:,:,st) = ...
                    log_p_jtp_stp_it_g_st_y1T(:,:,:,st) - log_p_st_g_y1T;
            end

            logl(st,t) = log_p_st_g_y1T;
        end

        % Collapse to a mixture of J Gaussians
        for st=1:S
            if logl(st,t) ~= -inf
                len = S*I*J;

                coeff = zeros(len,1);
                mean  = zeros(X,len);
                cov   = zeros(X,X,len);

                nmix = 0;

                for it=1:I
                    for stp=1:S
                        for jtp=1:J
                            tmp = log_p_jtp_stp_it_g_st_y1T(jtp,stp,it,st);

                            if tmp ~= -inf
                                nmix = nmix + 1;

                                coeff(nmix)   = exp(tmp);
                                mean(:,nmix)  = m_xt_y1T(:,jtp,stp,it,st);
                                cov(:,:,nmix) = Cov_xt_y1T(:,:,jtp,stp,it,st);
                            end
                        end
                    end
                end

                [newcoeff, newmean, newcov] = ...
                    mix2mix(coeff(1:nmix), mean(:,1:nmix), cov(:,:,1:nmix), J);

                L = length(newcoeff);

                logbpmix(1:L,st,t) = mylog(newcoeff);
                mu_smooth(:,1:L,st,t)      = newmean;
                Sigma_smooth(:,:,1:L,st,t)    = newcov;
            end % end if
        end % end st loop
    end % end t loop

    weights = exp(logl);            % Weights of the latent force models

    for t = 1:T
        for st=1:S
            % Smoothed distribution for each latent force model 
            mu_s(:,st,t) = exp(logbpmix(:,st,t))'*squeeze(mu_smooth(:,:,st,t))';
            for it = 1:J
                Sigma_s(:,:,st,t) = Sigma_s(:,:,st,t) + exp(logbpmix(it,st,t))*squeeze(Sigma_smooth(:,:,it,st,t));
            end

            % Overall smoothed distribution 
            Sigma(:,:,t) = Sigma(:,:,t) + weights(st,t)*squeeze(Sigma_s(:,:,st,t));
        end
        mu(:,t) = weights(:,t)'*squeeze(mu_s(:,:,t))';
    end
end
end