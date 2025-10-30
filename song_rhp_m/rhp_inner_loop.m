function [QC, QE, mtC, delta, mt, uld, uv, Tw, d0start, knc, FViterations, FVrestarts] = rhp_inner_loop(dEend, Tsat, delta0)
%RHP_INNER_LOOP 
%   iterate equations for each FV from evaporator end to condenser end
global RI;  % wall, rotor
global X N Nc Na DX max_inner_iterations NUMZERO max_restarts;     % discretization

knc=N;   % first non-converged FV (from end)
QE=0;    % heat flow in Evaporator (<0)
QC=0;    % heat flow in Condenser (>0)

FViterations=zeros(N-1,1);  % count iterations (per FV)
FVrestarts=zeros(N-1,1);  % count iterations (per FV)
Tw=zeros(N-1,1); % inner wall temperature (per FV)
delta=zeros(N,1); % fluid film height (per node)
mt=zeros(N,1);    % mass flow over node (left to right)
uld=zeros(N,1);    % mass flow over node (left to right)
uv=zeros(N,1);    % mass flow over node (left to right)

diffd0=diff(delta0);
delta(N)=dEend;   % prescribed and thus determining volume of liquid
mt(N)=0;  % no flow though end plate
uld(N)=0; % no liquid velocity at end plate
uv(N)=0;  % no vapour velocity at end plate
k=N;
while k>1   % E -> C instead of for-loop in order to allow for early termination
    d1=delta(k);
    mt1=mt(k);
    uld1=uld(k);
    uv1=uv(k);
    if k>=Nc+Na % evaporator
        d0start= max(d1-diffd0(k-1), delta0(k-1));  % diff=d1-d0
        [d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count] = iterate_fv_evaporator(d0start, d1, mt1, uv1, uld1, Tsat, k);
        Tw(k-1)=Twi; % inner wall temp. per FV
        QE=QE+qw*2*pi*RI(k-1)*DX(k-1);
    elseif k>Nc % adiabatic
        d0start=d1;
        [d0, mt0, uv0, uld0, Twi, ~, inner_iteration_count, restart_count] = iterate_fv_adiabatic(d0start, d1, mt1, uv1, uld1, Tsat, k);
        Tw(k-1)=Twi; % some value which does not distort the diagram
    else % condenser
        d0start=min(delta0(k-1)*delta(Nc)/delta0(Nc), d1);
      [d0, mt0, uv0, uld0, Twi, qw, inner_iteration_count, restart_count] = iterate_fv_condenser(d0start, d1, mt1, uv1, uld1, Tsat, k);
        Tw(k-1)=Twi; % inner wall temp. per FV
        QC=QC+qw*2*pi*RI(k-1)*DX(k-1);
    end % if CAE
    FViterations(k-1)=inner_iteration_count;
    FVrestarts(k-1)=restart_count;
    delta(k-1)=d0;
    mt(k-1)=mt0;
    uld(k-1)=uld0;
    uv(k-1)=uv0;    % for next FV
    
    % pick mass flow at condenser end (should be zero)
    if k==2
        mtC=mt0;
    end
    
    % if inner iterations do not converge..
    if ((inner_iteration_count==max_inner_iterations) || (restart_count==max_restarts))
        if (k>2) && (k<N)
            dmt=mt(k+1)-mt(k);     % use converged values (prev.FV) for extrapolation
            if (abs(dmt)>NUMZERO) % prevent diff by zero
                mtC=mt0-X(k)*dmt/(X(k+1)-X(k)); % extrapolate mt at Condenser plate
            else % if there were div0
                mtC=mt0;
            end
        else % first or last FV
            mtC=mt0;
        end
        delta(k-1)=0;   % for diagram beauty only (non-converged)
        knc=k;  % store non-converged FV
        k=1;    % stop FV evaluations from this point
    end % if not converged
    
    k=k-1;
end % looping through FVs from E to C

if ((inner_iteration_count<max_inner_iterations) && (restart_count<max_restarts))
    knc=0;
end

end % of function



