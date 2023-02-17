function [F,L,Qc,H] = cf_nn_to_ss(t,magnSigma02, magnSigma12, lengthScale, N)
%% CF_PERIODIC_TO_SS - Convert periodic covariance functions to continuous state space
%
% Syntax:
%   [F,L,Qc,H,Pinf,dF,dQc,dPinf,params] = 
%           cf_periodic_to_ss (magnSigma2, lengthScale, period, N, valid)
%
% In:
%   magnSigma2  - Magnitude scale parameter (default: 1)
%   lengthScale - Distance scale parameter (default: 1)
%   period      - Repetition period (default: 1)
%   N           - Degree of approximation (default: 6)
%   valid       - If false, uses Bessel function (default: false)
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
%   This function converts the so-called canonical periodic covariance
%   function to a state space model. The covariance function is
%   parameterized as follows:
%
%     k(tau) = magnSigma2 exp(-2 [sin(pi*tau/period)]^2/lengthScale^2)
%
%   where magnSigma2 is the magnitude scale parameter, lengthScale the  
%   distance scale parameter and period the repetition period length.
%   The parameter N is the degree of the approximation (see the reference
%   for details).
%     This function takes the covariance function parameters as inputs and
%   outputs the corresponding state space model matrices. The state space
%   model is given as follows in terms of a stochastic differential
%   equation
%
%      df(t)/dt = F f(t) + L w(t),
%
%   where w(t) is a white noise process with spectral denisty Qc.
%   The observation model for discrete observations y_k of f(t_k) at 
%   step k is as follows 
%
%      y_k = H f(t_k) + r_k, r_k ~ N(0, R),
%
%   where r_k is the Gaussian measurement noise wit covariance R.
%
% References:
%   [1] Arno Solin and Simo Särkkä (2014). Explicit link between periodic 
%       covariance functions and state space models. In Proceedings of the 
%       Seventeenth International Conference on Artifcial Intelligence and 
%       Statistics (AISTATS 2014). JMLR: W&CP, volume 33.
%
% See also:
%   COV_PERIODIC, SPEC_PERIODIC
%
% Copyright:
%   2012-2014 Arno Solin
%
%  This software is distributed under the GNU General Public
%  License (version 3 or later); please refer to the file
%  License.txt, included with the software, for details.

%% Apply defaults

  % Check if magnSigm2 is given
  if nargin < 1 || isempty(magnSigma02), magnSigma02 = 1; end
  if nargin < 1 || isempty(magnSigma12), magnSigma12 = 1; end
  % Check if lengthScale is given
  if nargin < 2 || isempty(lengthScale), lengthScale = ones(1,2*N); end 
  % Check if N is given
  if nargin < 4 || isempty(N), N = 6; end  
  
  
  
%% Form state space model  
  % The model
  F = zeros(2*N,2*N);
  for i = 1:N
    F(2*i-1:2*i,2*i-1:2*i) = [0 exp([1 t]*sqrt(diag(magnSigma02,magnSigma12))*[lengthScale(2*i-1); lengthScale(2*i)]); 0 0];
  end
  L    = eye(2*(N));
  Qc   = zeros(2*(N));
  H    = kron(ones(1,N),[1 0]);  
end

  