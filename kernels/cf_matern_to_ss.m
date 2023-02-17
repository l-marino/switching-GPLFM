function [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern_to_ss(magnSigma2, lengthScale, nu, N)
% CF_MATERN_TO_SS - Matern covariance functions to state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = cf_matern_to_ss(lengthScale, magnSigma2, nu, N)
%
% In:
%   magnSigma2  - Magnitude scale parameter (default: 1)
%   lengthScale - Length scale parameter (default: 1)
%   nu          - Matern smoothness parameter (default: 5/2)
%   N           - Approximation degree when nu is infinity (default: 6)
%
% Out:
%   F           - Feedback matrix
%   L           - Noise effect matrix
%   Qc          - Spectral density of white noise process w(t)
%   H           - Observation model matrix
%   Pinf        - Covariance of the stationary process
%   dF          - Derivatives of F w.r.t. parameters
%   dQc         - Derivatives of Qc w.r.t. parameters
%   dPinf       - Derivatives of Pinf w.r.t. parameters
%   params      - Input and output parameter information
%
% Description:
%   This function converts one-dimensional covariance functions of
%   the Matern class to state space models. The covariance function
%   parametrization is as follows
%
%     k(tau) = magnSigma2 2^(1-nu)/Gamma(nu) (sqrt(2 nu) |tau|/lengthScale)^nu
%              K_nu(sqrt(2nu) |tau|/lengthScale),
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter, nu the smoothness scale parameter and tau 
%   the time difference between states, tau = t-t'. K_nu is the modified 
%   Bessel function.
%     This function takes the covariance function parameters as inputs and
%   outputs the corresponding state space model matrices. The state space
%   model is given as follows in terms of a stochastic differential
%   equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc. The
%   observation model for discrete observation y_k of f(t_k) at step k, 
%   is as follows 
%
%      y_k = H f(t_k) + r_k, r_k ~ N(0, R),
%
%   where r_k is the Gaussian measurement noise with covariance R.
%     Pinf is the stationary covariance, where the value of Pinf(i,j), 
%   is defined as follows
%   
%      Pinf(i,j) = E[(f_i(t)-E[f_i(t)])(f_j(t)-E[f_j(t)])],
%
%   where f_i(t) is component i of state vector f(t).
%     Derivatives: All have same form. For example, dF has the following
%   form:
%
%       dF(:,:,1) = dF/d(magnSigma2 = input parameter_1),
%       dF(:,:,i) = dF/d(input parameter_i).
%
% References:
%
%   [1] Simo Sarkka, Arno Solin, Jouni Hartikainen (2013).
%       Spatiotemporal learning via infinite-dimensional Bayesian
%       filtering and smoothing. IEEE Signal Processing Magazine,
%       30(4):51-61.
%
%   [2] Jouni Hartikainen and Simo Sarkka (2010). Kalman filtering and 
%       smoothing solutions to temporal Gaussian process regression 
%       models. Proceedings of IEEE International Workshop on Machine 
%       Learning for Signal Processing (MLSP).
%
% See also:
%   COV_MATERN, SPEC_MATERN
%
% Copyright:
%   2012-2014   Arno Solin
%   2013-2014   Jukka Koskenranta
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma2), magnSigma2 = 1; end
  
  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = 1; end
  
  % Check if nu is given
  if nargin < 3 || isempty(nu), nu = 5/2; end
  
  % Check if N is given
  if nargin < 4 || isempty(N), N = 6; end
  
  % Case squared exponential (nu == inf)
  if isinf(nu)
      
      [F,L,Qc,H,Pinf,dF,dQc,dPinf] = cf_se_to_ss(magnSigma2, lengthScale, N);
    
  % Form state space model
  else
      
      % Number of dimensions
      d=1;            
      
      % Derived constants
      lambda = sqrt(2*nu)/lengthScale;
      
      % The process noise spectral density
      q = 2^d*pi^(d/2)*magnSigma2*gamma(nu+d/2)/gamma(nu)*lambda^(2*nu);
      
      % Compose dynamic model matrix
      p    = nu+d/2;
      Ni   = ceil(p);
      k    = 0:Ni;
      Ft   = bsxfun(@power,lambda*ones(1,Ni),p-k(1:end-1));
      coef = gamma(p+1)./factorial(k)./gamma(p-k+1);
      
      % Normalize
      Qc = q * (gamma(p+1)/factorial(Ni)/gamma(p-Ni+1))^(-2);
      coef = coef * (gamma(p+1)/factorial(Ni)/gamma(p-Ni+1))^(-1);
      
      % Feedback matrix
      F = diag(ones(Ni-1,1),1);
      F(end,:) = -bsxfun(@times,Ft,coef(1:end-1));
      
      % Noise effect matrix
      L  = [zeros(Ni-1,1);1];
      
      % Observation model
      H = zeros(1,Ni); H(1) = 1;
      
      % Stationary covariance
      
      % Calculate Pinf only if requested
      if nargout > 4,

          % Solve numerically as a solution to the algebraic Riccati equation
          % Pinf = are(F',zeros(size(F)),L*Qc*L');
          
          % Empirical analytic solution for Pinf, works only for nu =[1/2,3/2,5/2,7/2]
          Pinf = zeros(Ni);
          for k = 1:Ni
              for j = 1:Ni
                  if(mod(j+k,2)==0)
                      Pinf(j,k)=lambda^(j+k-2);
                      if(j~=k && all(abs(j-k) ~= 4:4:Ni))
                          Pinf(j,k)=-Pinf(j,k);
                      end
                  end
              end
          end
          Pinf = Pinf*magnSigma2/abs(2*(nu-1));
          Pinf([1,end])=Pinf([1,end])*abs(2*(nu-1));
      end
      
      % Calculate derivatives
      
      % Calculate derivatives only if requested
      if nargout > 5
          
          % Derivative of F w.r.t. parameter magnSigma2
          dFmagnSigma2 = zeros(size(F));
          
          % Derivative of F w.r.t parameter lengthScale
          dFlengthScale = zeros(size(F));
          dFlengthScale(end, :) = F(end,:)/lengthScale.*(-size(F,2):1:-1);
          
          % Derivative of Qc w.r.t. parameter magnSigma2
          dQcmagnSigma2 = Qc/magnSigma2;
          
          % Derivative of Qc w.r.t. parameter lengthScale
          dQclengthScale = Qc*(-2*nu)/lengthScale;
          
          % Derivative of Pinf w.r.t. parameter magnSigma2
          dPinfmagnSigma2 = Pinf/magnSigma2;
          
          % Derivative of Pinf w.r.t. parameter lengthScale
          lp=size(Pinf,1);
          coef = bsxfun(@plus,1:lp,(1:lp)')-2;
          coef(mod(coef,2)~=0)=0;
          dPinflengthScale = -1/lengthScale*Pinf.*coef;
          
          dF(:,:,1) = dFmagnSigma2;
          dF(:,:,2) = dFlengthScale;
          dQc(:,:,1) = dQcmagnSigma2;
          dQc(:,:,2) = dQclengthScale;
          dPinf(:,:,1) = dPinfmagnSigma2;
          dPinf(:,:,2) = dPinflengthScale;
          
      end
  end
  

%% Return parameter names
  
  % Only return if requested
  if nargout > 8
      
      % Input parameter information
      pa.in{1}.name = 'magnSigma2';   pa.in{1}.default = 1;   pa.in{1}.opt = true;
      pa.in{2}.name = 'lengthScale';  pa.in{2}.default = 1;   pa.in{2}.opt = true;
      pa.in{3}.name = 'nu';           pa.in{3}.default = 5/2; pa.in{3}.opt = false;
      pa.in{4}.name = 'N';            pa.in{4}.default = 6;   pa.in{4}.opt = false;
      
      % Return parameter setup
      params = pa;
      
  end
  