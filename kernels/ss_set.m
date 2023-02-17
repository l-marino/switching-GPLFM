function model = ss_set(model)
% SS_SET - Fills undefined parameters to model
%
% Syntax:
%   model = ss_set(model)
%
% In:
%   model - Stucture of state space model
%
% Out:
%   model - Model with parameters filled in
%
% Description:
%
% See also:
%   SS_UNPAK, SS_PAK, SS_STACK
%
% Copyright:
%   2013 Jukka Koskenranta 
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

%% Check the right form of model
        
  % Check if any ss model given
  if nargin < 1 || ~isfield(model,'ss') || isempty(model.ss); 
    warning('No state-space model given, using one squared exponential');
    model.ss{1}.make_ss=@cf_se_to_ss; 
  end;
  
  
%% Fill the model
    
  % Set sigma2
  if ~isfield(model,'sigma2')
    model.sigma2 = 1; % Default value
  end
  
  % Set the optimization status for sigma2
  if isfield(model, 'opt') && ...
     (isequal(model.opt,0) || isequal(model.opt,false));
      model.opt = false;
  else
      model.opt = true;
  end;
  
  % Set all model parameters and optimization status
  for j = 1: length(model.ss)
      
      % Check if ss former function is given
      if  isempty(model.ss{j}) || ...
         ~isfield(model.ss{j},'make_ss') || ...
         ~isa(model.ss{j}.make_ss, 'function_handle')
     
          warning(['No ss model given to model.ss{' num2str(j) '}, using square exponential instead']);
          model.ss{j}.make_ss = @cf_se_to_ss;
          
      end;
      
      % Set other parameters and optimization status for those
%       opts={};
% %       [foo1,foo2,foo3,foo4,foo5,foo6,foo7,foo8,params] = ...
%           [foo1,foo2,foo3,foo4] = ...
%           feval(model.ss{j}.make_ss);
%       
%       for k = 1:length(params.in)
%           
%           % Check if the value for this parameter is given
%           if ~isfield(model.ss{j}, params.in{k}.name)
%               model.ss{j}.(params.in{k}.name) = params.in{k}.default;
%           end
%           
%           % Check if optimization status defined for parameter in hand
%           if params.in{k}.opt && (~isfield(model.ss{j}, 'opt') || ...
%              any(strcmpi(params.in{k}.name, model.ss{j}.opt)))
%          
%               opts{numel(opts)+1} = params.in{k}.name;
%               
%           end
%       end
%       model.ss{j}.opt = opts;
  end