% To converge or not, this is the question in a single finite volume!
clear; close all; clc;

global kl hfg  rhol mul  nul betal cpl Pr;   % liquid
global rhov muv nuv;    % vapour
global kw Ro Ri RI omega alpha TC TE;  % wall, rotor
global X DX Lc La Le Riae;
global N Nc Na Ne mtC_rel_tol dmt_diff_rel_tol max_inner_iterations max_outer_iterations NUMZERO max_restarts MOD4GEOM;     % discretization

[~]=set_global_variables(1);

% specify FV
k=N; %N;
mt1=0; %0; 2.9198e-05
d1=0; %0; 1.9605e-05
d0start=1e-4;
uv1=0; %0.0; 0.13269
U1=0; %0.0; 0.07461
Tsat=80;

%% parameters from main_song.m
    %% EVAPORATOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%TODO COPY&PASTE ITERATE_EVAPORATOR
dx=DX(k-1); % index of FV
a=alpha(k-1); % index of FV
ri=Ri(k-1)/2 + Ri(k)/2; % mid-point values (x0/2+x1/2)
ro=Ro(k-1)/2 + Ro(k)/2; % mid-point values (x0/2+x1/2)
ri0=Ri(k-1); % index of left node (x=x0)
ri1=Ri(k); % index of left node (x=x0)
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
      
    %[dmt_ht, Tw, qw]=heat_transfer_model_film_condensation( d0,d1,TE,Tsat, dx,ri,ro, a );
    %[dmt_lf, b_lf]=liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv1,U1, Tw, Tsat );
    [dmt_ht, Tw, qw]=heat_transfer_model_mixed_convection( d0,d1,TE,Tsat, dx,ri,ro, a );
    if U1==0 % at evaporator end U1=0 makes trouble, no Cfd and no Cfw value..
        bN=2; % ..so generate a guess, assume parabolic velocity profile..
        U0_ht=dmt_ht*(bN+1)/(2*pi*ri0*rhol*d1); % ..find U0 from heat transfer..
        Um=(1/2)*U0_ht;    % ..take average between U1=0 and U0_ht
        uvm=0; % neglect vapour in rightest FV
    else % let the velocity at the right end enter into the calculation for Cfd and Cfw
        Um=U1;
        uvm=uv1;
    end
    [dmt_lf, b_lf]=liquid_flow_model_mixed_convection( d0,d1,mt1, dx,a,ri0, uvm,Um, Tw, Tsat );
    
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
end % while, iteration per FVM

[uv0,U0,~] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b_lf); % this enters next FV   

 disp('debug_fvm.m:')
disp(['k=',num2str(k)]);
disp(['mt0=',num2str(mt0)]);
disp(['d0=',num2str(d0)]);
disp(['uv0=',num2str(uv0)]);
disp(['U0=',num2str(U0)]);

