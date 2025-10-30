function  [dmt, b] = liquid_flow_model_mixed_convection( d0,d1,mt1, dx,a,ri0, uv1, uld1, Twi, Tsat )
%MIXED_CONVECTION 
% solve liquid flow model for difference in mass flow (liquid) 
% evaluation at begin (x=x0) of FV using uv and uld from x1 
% this convection model is assumed for evaporator
% d0...delta left
% d1...delta right
% mt1...mass flow in (right)
% dx...FV length
% a...taper angle
% ri0...inner radius (left)
% uv1...vapour velocity (right)
% uld1...liquid velocity at vapour interface (right)
% right side velocities determine characteristic numbers and thus flow regime

global rhol  mul nul Pr;   % liquid
global rhov nuv;    % vapour
global omega;  % wall

rim=ri0+(dx/2)*sin(a);
ri1=ri0+dx*sin(a);
dTm=Tsat-Twi;
deltam=(d0+d1)/2;
Av=pi*(rim-deltam)^2;  % cross-sectional area for vapour flow
dmt=0;          % initial guess, only needed to estimate liquid velocities (remember: dmt=mt1-mt0)
b_forced=2;     % assume forced convection for estimating u_avg
[~,~,u_avg] = liquid_velocities(d0, d1, mt1+dmt, dmt, ri0, dx, a, b_forced);

% characteristic numbers (fixed values at mid-point, iterated values right)
Gr=grashof(rim, a, dTm, deltam);  
Re=reynolds(deltam, u_avg); % u_avg is at x1 
Ra=Gr*Pr;
GrRe2=Gr/(Re^2);
% velocity profile power law
if Ra<1e9
    if GrRe2<0.1      % forced convection
        b=2;
    else
        if GrRe2<10     % mixed convection
            b=1/3;
        else
            b=1/4;
        end
    end
else
    b=1/7;      % natural convection
end

% wall friction [Song -> Afzal & Hussain]
if Re>0
    K=Gr/(Re^(5/2));
    if K<1
        Cfw=0.5;
    else
        Cfw=0.5*K^(3/5);
    end
else
    Cfw=0;
end

% vapour friction [Daniels]
Rev=uv1*Av/nuv;
if Rev>0
    if Rev<2000
        Cfd=16/Rev;
    else
        Cfd=0.0791/(Rev^0.25);
    end
else
    Cfd=0;
end

% top velocity from mass flow at x=x1
if d1>0
    U1=mt1*(b+1)/(2*pi*ri1*rhol*d1);
else
    U1=0; % no liquid, no velocity
end
 
% momentum bilance
QP = 0.4e1 * ((b + 1) * Av ^ 2 * ((Cfw * dx * b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + (Cfw * dx) / 0.2e1) * rhov + 0.4e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2) * U1 / (0.2e1 * (b + 1) * ((Cfw * dx * b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + (Cfw * dx) / 0.2e1 + 0.2e1 * deltam) * Av ^ 2 * rhov + 0.8e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2);
QQ = (-0.16e2 * rim * Av ^ 2 * omega ^ 2 * ((b + 1) ^ 2) * (b + 0.1e1 / 0.2e1) * rhov * (-d1 + d0) * deltam * cos(a) - 0.16e2 * rim * Av ^ 2 * omega ^ 2 * dx * ((b + 1) ^ 2) * (b + 0.1e1 / 0.2e1) * rhov * deltam * sin(a) + 0.2e1 * (Av ^ 2 * (Cfw * dx * (b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + Cfw * dx / 0.2e1 - 0.2e1 * deltam) * (b + 1) * rhov + 0.4e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2) * U1 ^ 2) / (0.2e1 * (b + 1) * (Cfw * dx * (b ^ 2) + (0.3e1 / 0.2e1 * Cfw * dx - d0 + d1) * b + Cfw * dx / 0.2e1 + 0.2e1 * deltam) * Av ^ 2 * rhov + 0.8e1 * Cfd * rim ^ 2 * rhol * dx * (b + 0.1e1 / 0.2e1) * pi ^ 2 * deltam ^ 2);


DISCRIMI=(QP/2)^2-QQ;
if DISCRIMI>=0
    U0=(-QP/2)+sqrt(DISCRIMI); % prefer higher value (TODO check)
else
    U0=0;
    disp('No real solution for U0 in liquid_flow_model_mixed_convection (evaporator).');
end

% % top velocity from mass flow at x=x1 and mass flow difference
mt0=U0*(2*pi*ri0*rhol*d1)/(b+1);
dmt=mt1-mt0;

%% debugging stuff


U0=0.07461;   % result from film condensation

% debug info
tauv = Cfd * rhol ^ 2 * pi ^ 2 * rim ^ 2 * (U0 + U1) ^ 2 * deltam ^ 2 / rhov / ((b + 1) ^ 2) / Av ^ 2 / 0.2e1;
tauw = Cfw * rhol * (U0 + U1) ^ 2 / 0.8e1;
Il = omega ^ 2 * (sin(a) * dx + (-d1 + d0) * cos(a)) * rim * deltam / dx;
Ir = -((U0 + U1) * ((-d1 + d0) * (U0 + U1) * b - 2 * deltam * (U0 - U1)) / dx / (2 * b ^ 2 + 3 * b + 1)) / 0.4e1;

%  disp(['lfmmc.m: Gr=',num2str(Gr),'   Ra=',num2str(Ra),'   Re=',num2str(Re)]);
%  disp(['lfmmc.m: Cfw=',num2str(Cfw),'   Cfd=',num2str(Cfd)]);
%  disp(['lfmmc.m: tauv/rhol=',num2str(tauv/rhol)]);
%  disp(['lfmmc.m: tauw/rhol=',num2str(tauw/rhol)]);
%  disp(['lfmmc.m: Il=',num2str(Il)]);
%  disp(['lfmmc.m: Ir=',num2str(Ir)]);


    
