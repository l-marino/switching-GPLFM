function [F,L,Qc,H,P0,dF,dQc,dPinf,dR,Hs] = ss_stack(model, pnames)  
% SS_STACK - Sum of state space models
%
% Syntax:
%   [F,L,Qc,H,P0,dF,dQc,dPinf,dR,Hs] = ss_stack(model, pnames)
%
% In:
%   model  - Stucture of (multiple) state-space model(s)
%   pnames - Locations for optimized parameters (optional)   
%
% Out:
%   F      - Feedback matrix for superposition model
%   L      - Noise effect matrix for superposition model
%   Qc     - Spectral density of the white noise process 
%   H      - Observation model matrix for superposition model
%   Pinf   - Covariance of the stationary process for superposition model
%   dF     - Derivatives of F w.r.t. parameters for superposition model
%   dQc    - Derivatives of Qc w.r.t. parameters for superposition model
%   dPinf  - Derivatives of Pinf w.r.t. parameters for superposition model
%   Hs     - Componentwise observation model matrix 
%
% See also:
%   SS_PAK, SS_UNPAK, SS_SET
%
% Copyright:
%   2012-2013 Arno Solin
%   2013      Jukka Koskenranta
%
% This software is distributed under the GNU General Public
% License (version 3 or later); please refer to the file
% License.txt, included with the software, for details.

%%

  F   = [];
  L   = [];
  Qc  = [];
  H   = [];
  P0  = [];
  dF  = [];
  dQc = [];
  dPinf = [];
  Hs    = [];
  
  % Stack state space models
  for j=1:numel(model.ss)    
      
    [foo1,foo2,foo3,foo4,foo5,foo6,foo7,foo8,params] = ...
        feval(model.ss{j}.make_ss);
    
    % Input parameters in one stucture vector theta
    theta = arrayfun(@(z) {model.ss{j}.(params.in{z}.name)}, ...
        (1:length(params.in)));

    % SS former function
    make_ss=model.ss{j}.make_ss;
    
    % Form ss model j
    if nargout > 5
        [jF,jL,jQc,jH,jP0,jdF,jdQc,jdPinf] = make_ss(theta{:});
    else
        [jF,jL,jQc,jH,jP0] = make_ss(theta{:});
    end
    
    % Stack matrices
    F  = blkdiag(F,jF);
    L  = blkdiag(L,jL);
    Qc = blkdiag(Qc,jQc);
    H  = [H jH];    
    P0 = blkdiag(P0,jP0);
    
    % Stack derivative matrices
    if nargout > 5
        
      % Number of patrial derivatives (except R)
      njder = min(size(jdF,3),numel(jdF));
      
      % Check if pnames given
      if nargin < 2 || isempty(pnames)
        idd = true(1,njder);
      else        
        % Id for used derivatives (which parameters are used)
        idd = arrayfun(@(k) any(strcmpi(params.in{k}.name, ...
            pnames([pnames{:,1}]==j,2))),1:length(params.in));
      end

      % Add chosen derivatives
      dF  = mblk(dF,jdF(:,:,idd));
      dQc = mblk(dQc,jdQc(:,:,idd));
      dPinf = mblk(dPinf, jdPinf(:,:,idd));
      
    end
    
    % If requested, stack the observation models for separate outputs
    if nargout>9, Hs = blkdiag(Hs,jH); end
    
  end
  
  % Add derivatives w.r.t sigma2
  % Expects that sigma2 is the first optimized parameter if it is optimized
  if nargout > 5 
      % Number of patrial derivatives (except R)
      nder = min(size(dF,3),numel(dF));
      
      % Derivatives of measurement noise variance
      dR = zeros(1,1,nder);
      if nargin < 2 || isempty(pnames) || pnames{1,1} == 0
          
          if nargin > 1 && ~isempty(pnames) && ...
             ~isequal(size(pnames,1),nder+1)
              error('Number of patrial derivatives does not match number of optimized parameters')
          end
          
          dR = zeros(1,1,nder+1);
          dR(1)=1;
          
          % Allocate space for new derivatives fd*
          fdF = zeros([size(F,1),size(F,2), nder+1]);
          fdQc = zeros([size(Qc,1),size(Qc,2), nder+1]);
          fdPinf = zeros([size(P0,1),size(P0,2), nder+1]);
          
          % Assign old d* to new fd* if old one exists
          if ~isempty(dF),
            fdF(:,:,2:end) = dF; 
            fdQc(:,:,2:end) = dQc; 
            fdPinf(:,:,2:end) = dPinf; 
          end;
          
          % Replace the old d* with the old one
          dF = fdF; dQc =fdQc; dPinf = fdPinf;
          
      elseif nargin > 1 && ~isempty(pnames) && ...
              ~isequal(size(pnames,1),nder)
          error('Number of patrial derivatives does not match number of optimized parameters')
      end
  end

end


function C = mblk(A,B)
% 3 dimensional version of blk function

  % Get sizes
  sA=size(A); sB=size(B);

  % Check if A or B is empty
  Ae = ~any(sA==0); Be = ~any(sB==0);
  
  % Numel of sizes to 3
  sA = [sA Ae]; sA = sA(1:3);
  sB = [sB Be]; sB = sB(1:3);
  
  % Assign space for C
  C = zeros(sA+sB);
  
  % Set values of A if there is any
  if Ae
      C(1:sA(1), 1:sA(2), 1:sA(3)) = A;
  end
  
  % Set values of B if there is any
  if Be
      C(sA(1)+(1:sB(1)), ...
        sA(2)+(1:sB(2)), ...
        sA(3)+(1:sB(3))) = B;
  end

end


