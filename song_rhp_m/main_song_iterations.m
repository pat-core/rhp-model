% Model of ROTATING HEAT PIPE [Song2003]
% condenser-adiabatic-evaporator (left to right)
% SI units (m,kg,K,s)
%
% 1D-FV discretization:
% condenser and adiabatic section modeled by FILM-CONDENSATION MODEL
% evaporator modeled by MIXED CONVECTION MODEL 
% iteration variables (nodal values):
% delta...liquid height (CAE)
% 1) liquid flow model delta -> dmt_lf
% 2) heat transfer model delta -> dmt_ht (implicitely includes Twi)
% 3) iterate delta from E to C until dmt_lf=dmt_ht in each FVM
% 4) outer loop iterates Tsat until QC=QE (implicitely fulfils mt(Cbegin)=0)
%
% TODO: 
%	debug mixed convection
%   pass operational parameters omega,TE,TC (non-global)
%	code cosmetics, such as unify variable names over functions (d/delta, U/uld, inner iterations -> FViterations, outer iterations -> grid points)
%	consider dry-out via reduction of Le
%
% TODO main does reproduce the results of Song in the following order
%   1) changing delta_{N+1}
%   2) changing Te-Tc
%   3) changing RPM
% 
% PREPARE FOR PUBLIC DISTRIBUTION (style & comment)

clear; clc; close all;
global kl hfg  rhol mul  nul betal cpl Pr;   % liquid
global rhov muv nuv;    % vapour
global kw Ro Ri RI omega alpha TC TE;  % wall, rotor
global X DX Lc La Le Riae;
global N Nc Na Ne mtC_rel_tol dmt_diff_rel_tol max_inner_iterations max_outer_iterations NUMZERO max_restarts MOD4GEOM;     % discretization

[~]=set_global_variables(1);

Nd=1; % 21 %discretization levels 
% main operational parameters
%dEend2D=0.0;    % fluid height/diameter at evaporator end (=0 -> minimal fluid loading)
dEend2Dmin=0.000;
dEend2Dmax=0.0058595; % 0.035 (this value is chosen for Nd=1)

meanRi=mean(Ri);
meanRic=mean(Ri(1:Nc));

L=Lc+La+Le; % total length
Di=2*Riae;


% initial guess for first run (next runs take previous results as guess)
%Tsat_prev=(1/2)*(TC+TE); % initial guess for vapour temperature
dEendmin=(2*meanRi)*dEend2Dmin; % minimal value of fluid height at E (typically zero))
ml=look_up_song(dEend2Dmin);    % look up from [Song2003]
delta_konst=((ml/rhol)/(2*pi*meanRi)-dEendmin*Le/2)/(Lc/2+La+Le/2);
dc0=linspace(0,delta_konst,Nc).';
da0=ones(Na-2,1)*delta_konst;   % because first and last node are in dc/de
de0=linspace(delta_konst,dEendmin,Ne).';
delta0=[dc0; da0; de0];
%V0=liquid_volume(delta0,RI);
%m0=rhol*V0;


%% run numerical solution (loop: dEend, Tsat, rhp-FVs, FV-dmt)
disp(['-- RUNNING film condensation/evaporation model with N=',num2str(N),' finite volumes --']);
fileID = fopen('local_iterations.log','w');

dEend2Dinput=linspace(dEend2Dmin,dEend2Dmax,Nd); % vector of dEend2D values
delta_results=zeros(N,Nd);  % nodal
mt_results=zeros(N,Nd);     % nodal
GrRe2_results=zeros(N-1,Nd);     % FV
Tw_results=zeros(N-1,Nd);    % FV
qc_results=zeros(Nd,1);   % heat flux through inner condenser wall
V_results=zeros(Nd,1);    % total liquid volume
Tsat_results=zeros(Nd,1);    % total liquid volume
Tsat0=(TC+TE)/2;  % in order to have a maximum for Tsat_range in first run

for kk=1:Nd % loop film heigths at evaporator end
    
    % changing parameter
    dEend2D=dEend2Dinput(kk);  
    dEend=(2*meanRi)*dEend2D;
    disp(' ');
    disp(['dEend/D=', num2str(dEend2D)]);
    fprintf(fileID, 'dEend/D=%6.6f \n', dEend2D);   

    [delta, mt, Tw, GrRe2, V, Tsat_ss, qc, knc, count_converged, Tsat_v, mtC_rel_v, QCE_rel_v, index_converged, index_diverged]= rhp_outer_loop(dEend, Tsat0, delta0, fileID);
    
    % store results for one value of dEend
    delta_results(:,kk)=delta;
    mt_results(:,kk)=mt;
    Tw_results(:,kk)=Tw;
    GrRe2_results(:,kk)=GrRe2;
    V_results(kk)=V;
    Tsat_results(kk)=Tsat_ss;
    qc_results(kk)=qc;
    fprintf(fileID, '\n');
    
    if (knc>0) || (count_converged<2)
        disp('_');
        disp(['WARNING, solution not converged for DEend/D=', num2str(dEend2D)]);
        disp(['count_converged=',num2str(count_converged),';']);
        disp('DEBUG INFO'); disp(' ');
        disp(['knc=',num2str(knc),';']);
        if (knc>1) % copy & paste to debug.m
            disp(['mt1=',num2str(mt(knc),10),';']);
            disp(['d1=',num2str(delta(knc),10),';']);
%            disp(['d0start=',num2str(d0start,10),';']);
%            disp(['uv1=',num2str(uv(knc),10),';']);
%            disp(['uld1=',num2str(uld(knc),10),';']);
            disp(['Tsat=',num2str(Tsat_ss,10),';']); % value for final run
        end
        break;  % no hope to try higher values of dEend
    else
        %        disp(['deltaC/max(delta)=',num2str(delta(1)/max(delta))]);  % should be zero
        disp(['Tsat=',num2str(Tsat_ss),'°C   liquid mass m=', num2str(V*rhol*1000),' g','   heat flux qc=',num2str(qc),' W/m^2']);  % should be zero
    end
    
    delta0=delta;
    Tsat0=Tsat_ss;
end % for kk=1:Nd loop dEend

%% postprocessing 
figure;
plot(Tsat_v(index_converged),    QCE_rel_v(index_converged),   'r.',...
     Tsat_v(index_diverged), QCE_rel_v(index_diverged),'ro',...
     Tsat_v(index_converged),    mtC_rel_v(index_converged),   'b.',...
     Tsat_v(index_diverged), mtC_rel_v(index_diverged),'bo',...
     Tsat_ss,0, 'k*'); 
xlabel('T_{sat}'); ylabel('ERROR'); % (con-/divergence)
%title('last value for dEend: root finding for Tsat');
if count_converged>0
    legend('Q con', 'Q div','mt con', 'mt div','Tsat ss');
else
    legend('Q div','mt div','Tsat ss');
end
%matlab2tikz('tsat_search.tex', 'height', '\figureheight', 'width', '\figurewidth' );

% % evaluate last dEend
% Xm=(X(1:end-1)+X(2:end))/2; % for plotting mid-point values
% figure; % CAE
% subplot(221); plot(X/L, delta/(2*meanRi),'k.'); 
% xlabel('x/L'); ylabel('d/D'); title(['dEend/D=',num2str(dEend2Dinput(kk))]);
% if (knc>0)
%     hold on; plot(X(knc)/L, delta(knc)/(2*meanRi),'ro');
% end
% subplot(222); plot(X/L, mt); xlabel('x/L'); ylabel('mt');
% subplot(223); plot(Xm/L, Tw-Tsat_ss); xlabel('x/L'); ylabel('Twi-Tsat ss');
% subplot(224); semilogy(Xm(N-Ne+1:N-1)/L, GrRe2(N-Ne+1:N-1)); xlabel('x/L'); ylabel('Gr/Re^2'); xlim([X(1)/L, X(end)/L]);
% 
% figure;
% plot(dEend2Dinput, Tsat_results); xlabel('$\delta_{N+1}/D$','Interpreter','LaTeX'), ylabel('$T_\mathrm{sat}$','Interpreter','LaTeX');
% matlab2tikz('deltaend_tsat.tex', 'height', '\figureheight', 'width', '\figurewidth' );
% 
% figure;
% plot(dEend2Dinput, V_results/V_results(1)); xlabel('$\delta_{N+1}/D$','Interpreter','LaTeX'), ylabel('$V/V_\mathrm{min}$','Interpreter','LaTeX');
% matlab2tikz('deltaend_V.tex', 'height', '\figureheight', 'width', '\figurewidth' );
% 
% figure;
% plot(X/L,delta_results/(2*meanRi)); xlabel('$x/L$','Interpreter','LaTeX'); ylabel('$\delta/D$','Interpreter','LaTeX');
% matlab2tikz('x_delta.tex', 'height', '\figureheight', 'width', '\figurewidth' );
% 
% plot(Xm/L,Tw_results); xlabel('$x/L$','Interpreter','LaTeX'); ylabel('$T_\mathrm{wi}$','Interpreter','LaTeX')
% matlab2tikz('x_tw.tex', 'height', '\figureheight', 'width', '\figurewidth' );




