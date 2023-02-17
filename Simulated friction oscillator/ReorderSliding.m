function [v_ord, f_ord] = ReorderSliding(v,f,tol)
%--------------------------------------------------------------------------
% This function reorders the element of the friction force-velocity curve,
% removing incorrect estimates where friction and velocity have the same
% sign, friction force values around zero velocity (|v|<=tol) and outliers 
% (outside of 95% bounds).
% 
% Variables:
% v = velocity vector;
% f = friction force vector;
% tol = estimates with |v|<=tol will be disregarded
% 
% Output:
% v_ord, f_ord = sorted velocity and friction force vectors
%--------------------------------------------------------------------------

[v_ord,index] = sort(v);
f_ord = f(index);

v_gp_plus = v_ord(v_ord>tol & f_ord>=0);
f_gp_plus = f_ord(v_ord>tol & f_ord>=0);
v_gp_minus = -v_ord(v_ord<-tol & f_ord<=0);
f_gp_minus = -f_ord(v_ord<-tol & f_ord<=0);
v_ord = [v_gp_plus v_gp_minus];
f_ord = [f_gp_plus f_gp_minus];
[v_ord,index] = sort(v_ord);
f_ord = f_ord(index);

f_mean = mean(f_ord);
f_std = std(f_ord);

v_ord = v_ord(f_ord>min(f_ord(end-1),f_mean-2*f_std)&f_ord<(f_mean+2*f_std));
f_ord = f_ord(f_ord>min(f_ord(end-1),f_mean-2*f_std)&f_ord<(f_mean+2*f_std));    
end