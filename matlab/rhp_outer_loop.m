function [delta, mt, Tw, GrRe2, V, Tsat_ss, qc, knc, count_converged, Tsat_v, mtC_rel_v, QCE_rel_v, index_converged, index_diverged]= rhp_outer_loop(dEend, Tsat0, delta0, fileID)
%RHP_OUTER_LOOP
% iterate over Tsat until physical plausible state (no flow through wall)
% knc...first non-converged FV (from end) in last inner iteration
% count_converged...number of completely converged inner iterations
global hfg  rhol ;   % liquid

global RI alpha TC;  % wall, rotor
global N Nc DX mtC_rel_tol max_inner_iterations max_outer_iterations NUMZERO max_restarts;     % discretization

Ac=sum(2*pi*RI(1:Nc-1).*DX(1:Nc-1)); % for heat flux qc at condenser

%% generate Tsat-grid (minimum to maximal possible value)
    %Tsat_v=linspace(TC+NUMZERO,Tsat_ss-NUMZERO, max_outer_iterations).'; % Tsat_ss from previous run bounds meaningful range
    LOGN=(1-logspace(-2,0,max_outer_iterations))*100/99;
    Tsat_v=TC+NUMZERO+LOGN*(Tsat0-TC);    
    knc_v=zeros(max_outer_iterations,1); % store index of non-converged FV for each Tsat (allows to extract converged results)
    mtC_rel_v=zeros(max_outer_iterations,1); % for heat flow "through condenser wall" as criteria
    QCE_rel_v=zeros(max_outer_iterations,1); % for heat bilance as termination criteria

%% evaluate Tsat-grid    
for outer_iteration_count=1:max_outer_iterations   % evaluate heat pipe (all FVs) to find Tsat
    Tsat=Tsat_v(outer_iteration_count);
    [QC, QE, mtC, ~, mt, ~, ~, ~, ~, knc, ~, ~] = rhp_inner_loop(dEend, Tsat, delta0);
    QCE=(QC+QE);
    QCE_rel=QCE/(abs(QC)+abs(QE));
    QCE_rel_v(outer_iteration_count)=QCE_rel;   % for heat bilance as termination criteria
    mtC_rel=mtC/max(mt);
    mtC_rel_v(outer_iteration_count)=mtC_rel;   % for heat flow "through condenser wall" as criteria
    knc_v(outer_iteration_count)=knc;
end
% end for/while-loop for Tsat

%% determine Tsat
% inter/extrapolate steady state Tsat from converged results 
index_converged=find(knc_v==0);
index_diverged=find(knc_v>0);
count_converged=length(index_converged);
if count_converged>2
    Tsat_ss=interp1(mtC_rel_v(index_converged), Tsat_v(index_converged), mtC_rel_tol, 'spline');  % slightly positive zero (for non-positive value problems on last FV)
elseif count_converged>0
    Tsat_ss=interp1(mtC_rel_v(index_converged), Tsat_v(index_converged), mtC_rel_tol, 'linear','extrap');  % slightly positive zero (for non-positive value problems on last FV)
else
    Tsat_ss=min(Tsat_v(knc_v==min(knc_v))); % in order to have some values, take the run where k_non_converged is minimal
end

%% final run (interpolated Tsat)
[QC, QE, mtC, delta, mt, uld, ~, Tw, ~, knc, FViterations, FVrestarts] = rhp_inner_loop(dEend, Tsat_ss, delta0);
QCE=(QC+QE);
QCE_rel=QCE/(abs(QC)+abs(QE));
mtC_rel=mtC/max(mt);

% convergence info
fprintf(fileID, 'mtC_rel=%1.12f   QCE_rel=%1.12f   deltaC/max(delta)=%4.2f   QC=%4.2f QE=%4.2f   converged=%d/%d \n', mtC_rel, QCE_rel, delta(1)/max(delta), QC, QE, count_converged, max_outer_iterations);
fprintf(fileID, 'inner iterations (final run): mean=%4.2f   min=%d   max=%d    (max_inner_iterations=%d) \n', mean(FViterations(FViterations>0)), min(FViterations(FViterations>0)), max(FViterations), max_inner_iterations);
if (max(FVrestarts)>0)
    fprintf(fileID, 'inner restarts (final run): mean=%4.2f   min=%d   max=%d   (max_restarts=%d) \n', mean(FVrestarts(FViterations>0)), min(FVrestarts(FViterations>0)), max(FVrestarts), max_restarts);
else
    fprintf(fileID, 'inner restarts (final run): 0 restarts \n');
end

%% results
V=liquid_volume(delta, RI);
mt_max=max(mt);
Q=mt_max*hfg; % maximum flow occurs in adiabatic section end determines transfered heat
qc=Q/Ac;    % inner condenser wall
fprintf(fileID, 'Tsat_ss=%3.6f°C   ml=%3.6f g   max(mlt)=%3.6f kg/s   qc=%3.6f W/m^2 \n\n', Tsat_ss, V*rhol*1000, mt_max, qc);

% prepare plots to compare with Song
GrRe2=zeros(N-1,1);
for k=1:N-1 % per FV
    Um=(1/2)*(uld(k)+uld(k+1));
    deltam=(1/2)*(delta(k)+delta(k+1));
    Gr=grashof(RI(k),alpha(k),Tw(k)-Tsat_ss, deltam);
    Re2=(reynolds(deltam,Um))^2;
    if Re2>0
        GrRe2(k)=Gr/Re2;
    else
        GrRe2(k)=0;
    end
end

end

