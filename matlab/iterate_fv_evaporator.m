function [d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count] = iterate_fv_evaporator(d0start, d1, mt1, uv1, uld1, Tsat, k)
%EVAPORATOR
% k=1..N=Nc+Na+Ne (node numbering, there is one more node than FV)
global Ro Ri alpha TE;  % wall
global DX dmt_diff_rel_tol max_inner_iterations max_restarts;     % discretization

dx=DX(k-1); % index of FV
a=alpha(k-1); % index of FV
ri=Ri(k-1)/2 + Ri(k)/2; % mid-point values (x0/2+x1/2)
ro=Ro(k-1)/2 + Ro(k)/2; % mid-point values (x0/2+x1/2)
ri0=Ri(k-1); % index of left node (x=x0)
% restart correction factors for delta0
K=restart_values(max_restarts);     % last value not executed

% initialize FVM iteration
d0=d0start;
dmt_diff_rel=1; % set to enter iteration
mt0=-mt1;% set to enter iteration
restart_count=0;
iteration_count=0;

while ((dmt_diff_rel>dmt_diff_rel_tol) || (mt0<0)) && (restart_count<max_restarts) && (iteration_count<max_inner_iterations)
    iteration_count=iteration_count+1;
    
    %[dmt_ht, Tw, qw]=heat_transfer_model_film_condensation( d0,d1,TE,Tsat, dx,ri,ro );
    [dmt_ht, Tw, qw]=heat_transfer_model_mixed_convection( d0,d1,TE,Tsat, dx,ri,ro, a );
    %[dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,uld1 );
    [dmt_lf, b_lf]=liquid_flow_model_mixed_convection( d0,d1,mt1, dx,a,ri0, uv1,uld1, Tw, Tsat );
    
    dmt_diff_rel=abs(dmt_ht-dmt_lf)/(abs(dmt_ht)+abs(dmt_lf));
    
    fz=(dmt_ht-dmt_lf);   %log(abs(dmt_ht/dmt_lf)); % function to be iterated to zero
    if iteration_count == 1 % not enough values for regula falsi
        if fz>0
            d0next=(4/5)*d0; % increase to some value
        else
            d0next=(5/4)*d0; % decrease to some value
        end
    else % determine next value from current and previous ones
        d0next=d0-fz*(d0-d0prev)/(fz-fzprev); % regula falsi
    end
    
    % shift to next iteration
    fzprev=fz;  % function value (that should be eiterated to zero)
    d0prev=d0;  
    
    if (d0next<0) || isnan(d0next)
        restart_count=restart_count+1;
        if d0start==0
            d0=K(restart_count)*d1;
        else
            d0=K(restart_count)*d0start;
        end
        dmt=0; % in order to guarantee a return value
    else
        d0=d0next;
        dmt=dmt_ht; %dmt_lf/2+dmt_ht/2;    % this gets important if ht<>lf (not converged)
        mt0=mt1-dmt;    % mt0+dmt=mt1
    end
end % iteration per FVM

[uv0,uld0,~] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); % this enters next FV

end

