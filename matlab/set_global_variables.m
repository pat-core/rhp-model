function [set_flag] = set_global_variables(configuration)
%SET_GLOBAL_VARIABLES 
% right now there is only one config, which is of [Song2003]

global kl hfg  rhol mul  nul betal cpl Pr;   % liquid
global rhov muv nuv;    % vapour
global kw Ro Ri RI omega alpha TC TE;  % wall, rotor
global X DX La Lc Le Riae;
global N Nc Na Ne mtC_rel_tol dmt_diff_rel_tol max_inner_iterations max_outer_iterations NUMZERO max_restarts MOD4GEOM;     % discretization

% discretization parameters
%Nc1=5; Nc2=30; Nc3=5; Na=20; Ne=30;   % "adaptive" nodes CAE (89-57-76)
Nc=23; Na=10; Ne=19;   % nodes CAE (89-57-76) Nc=45; Na=28; Ne=38;
NUMZERO=1e-14;   % for iterations in adiabatic section
dmt_diff_rel_tol=1e-6; % iteration tol. for rel. difference in mass flow
max_outer_iterations=50;  % number Tsat mesh points (outer loop over all FV)
max_inner_iterations=100; % maximum number of iterations for mt (inner iterations for each FV)
max_restarts=38;    % restart iterations within inner loop (per FV)
MOD4GEOM=50;    % use geometric mean in between to accelerate convergence
mtC_rel_tol=1e-2;  % iteration tol. for mt at Condenser end corresponding to Tsat



TC=60;   % outer wall temperature (C)
TE=100;   % outer wall temperature (E)
rpm=20000; % revolutions per minute, determines OMEGA

% liquid and vapour parameters
kl=0.598;   % thermal conductivity water
kw=45;   % thermal conductivity steel
rhov=0.768; %0.768; % density vapour (p=101.300kPa, T=100°C)
rhol=998;  % density water
muv=12.4e-6;   % viscosity vapour (T=100°C, 1bar) http://www.wissenschaft-technik-ethik.de/wasser_eigenschaften.html#kap06
mul=1e-3;       % viscosity water (T=20°C)
betal=(1/3)*20.7e-5;  % thermal expansion
cpl=4186;   % spezific heat capacity
hfg=2.257e6;    % latent heat
% geometry
Lc=0.089; La=0.057; Le=0.076;   % lengths, determines XC,XA,XE
Ric=0.0160/2;    % inner radius at condenser (left), determines RI
Riae=0.0191/2;    % inner radius at adiabatic section (left), constant from there on, determines RI
Ro_cae=0.0254/2;   % outer radius, determines RO

% derived parameters
%Nc=Nc1+Nc2+Nc3-2; % number of nodes (concatination at C12 and C23)
N=Nc+Na+Ne-2;   % number of nodes (concatination at CA and AE)
%Xc1=linspace(        0, (1/10)*Lc, Nc1);
%Xc2=linspace((1/10)*Lc, (9/10)*Lc, Nc2);
%Xc3=linspace((9/10)*Lc, Lc       , Nc3);
%Xc=[Xc1, Xc2(2:end-1), Xc3(1:end)];
Xc=linspace(0, Lc, Nc); % determines X
Xa=Lc+linspace(0,La,Na); % determines X
Xe=Lc+La+linspace(0,Le,Ne); % determines X
X=[Xc, Xa(2:end-1), Xe(1:end)].';   
DX=diff(X);
nul=mul/rhol;   % kin. viscosity liquid
nuv=muv/rhov;   % kin. viscosity vapour
Pr=nul/(kl/(rhol*cpl));  % Prandtl Number
omega=(rpm/60)*2*pi; % rotational velocity
alpha=zeros(N-1,1); % taper angle
alpha(1:Nc-1)=atan(Riae-Ric)/Lc;
Ri=Ric+[0; cumsum(DX.*sin(alpha))];
Ro=ones(N,1)*Ro_cae;    % outer radius
RI=Ri(1:end-1)/2 + Ri(2:end)/2; % mid-value between nodes
RO=Ro(1:end-1)/2 + Ro(2:end)/2;


set_flag=1;
end

