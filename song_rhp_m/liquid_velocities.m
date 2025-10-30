function [uv0, uld0, u_avg0] = liquid_velocities(d0, d1, mt0, dmt, ri0, dx, a, b)
%LIQUID_VELOCITIES 
% velocity of vapour and liquid top layer
% at left side of given FVM (forward FD)
% a...taper angle alpha
% b...power law exponent
% uv0...vapour velocity at x=x0
% uld0...liquid velocity at vapour interface at x=x0
% u_avg0...average liquid velocity at x=x0
global rhol rhov;

% vapour velocity
uv0=mt0/(rhov*pi*ri0^2); % stationary: mass flow forward (liquid)=mass flow backward (vapour)

% liquid flow model (may contain pole)
%T0=(rhol/mul)*(sin(a)*cos(a)*(d1-d0)/dx)*(1/2)*d0^2;
%T1=(d0/mul)*(dmt/dx);
%uld0=(T0-T1*uv0*cos(a))/(1+T1);

% liquid velocity
Vt0=mt0/rhol;
u_avg0=Vt0/(2*pi*ri0*d0); % average
uld0=(b+1)*u_avg0; % vapour interface


end

