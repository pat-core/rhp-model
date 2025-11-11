function [d0, mt0, uv0, uld0, Tw, qw, iteration_count, restart_count] = iterate_fv_adiabatic(d0start, d1, mt1, uv1, uld1, Tsat, k)
%ADIABATIC
% k=1..N=Nc+Na+Ne (node numbering, there is one more node than FV)
global  Ri  alpha;  % wall
global  DX NUMZERO max_inner_iterations max_restarts;     % discretization

dx=DX(k-1); % index of FV
a=alpha(k-1); % index of FV
%ri=Ri(k-1)/2 + Ri(k)/2; % mid-point values (x0/2+x1/2)
%ro=Ro(k-1)/2 + Ro(k)/2; % mid-point values (x0/2+x1/2)
ri0=Ri(k-1); % index of left node (x=x0)

K=restart_values(max_restarts);

% initialize FVM iteration
d0=d0start;
iteration_count=0;
restart_count=0;

dmt_lf=1; % CHANGE compared to iterate_CE.m % enter iteration
dmt_ht=0; % CHANGE compared to iterate_CE.m

while (abs(dmt_lf)>NUMZERO) && (iteration_count<max_inner_iterations) && (restart_count<max_restarts)
    iteration_count=iteration_count+1;
    
    [dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,uld1 );
    
    fz=(dmt_ht-dmt_lf);   % function to be iterated to zero
    
    if iteration_count == 1 % not enough values for regula falsi
        if fz>0
            d0next=(4/5)*d0; % increase to some value
        else
            d0next=(5/4)*d0; % decrease to some value
        end
    else % determine next value from current and previous ones
        d0next=d0-fz*(d0-d0prev)/(fz-fzprev); % regula falsi
    end
    
    d0prev=d0;  % shift to next iteration
    if isnan(d0next) || d0next<0
        restart_count=restart_count+1;
        d0=K(restart_count)*d0start;
    else
        d0=d0next;
    end
    fzprev=fz;  % function value (that should be eiterated to zero)
    
end % iteration per FVM

dmt=dmt_ht;   % CHANGE compared to iterate_CE.m
mt0=mt1-dmt;    % mt0+dmt=mt1

[uv0,uld0,~] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); % this enters next FV

Tw=Tsat; % some value which does not distort the diagram
qw=0; % only for uniform return values

end

