function [w,pnames] = ss_pak(model)
% SS_PAK - Extract parameters from model to vector
%
% Syntax:
%   [w,pnames] = ss_pak(model)
%
% In:
%   model       - Stucture of state-space model
%
% Out:
%   w           - Initial values for optimized parameters
%   pnames      - Locations of optimized parameters in model 
%
% Description:
%   Expects that the model has not been changed after using 
%   model=ss_set(model). Preferred use is to generate initial values w0 
%   and 'pnames' for the parameters.
%
% See also:
%   SS_UNPAK, SS_SET, SS_STACK
%
% Copyright:
%   2013 Jukka Koskenranta 
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

%%

  % Initialize the missing fields
  model = ss_set(model);

  % Model measurement noise variance
  if model.opt
    w = model.sigma2;
    pnames(1,:) = {0, 'sigma2'};
  else
    w = [];
    pnames = {};
  end

  % Other parameters (for each model in the model structure) 
  for j = 1:length(model.ss)
   
    ss = model.ss{j};
    opts = ss.opt;

    % All parameters in model j
    for k = 1:length(opts)
       w = [w ss.(opts{k})];
       pnames(numel(w),:) = {j,opts{k}};
    end
   
  end
