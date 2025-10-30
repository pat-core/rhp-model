function  [dmt, b] = liquid_flow_model_film_condensation( d0,d1,mt1, dx,a,ri0, uv,uld, ~,~ )
%FILM_CONDENSATION
% solve liquid flow model for difference in mass flow (liquid) 
% evaluation at begin x=x0 (left) of FV using uv and uld from x1 (right)

global rhol  mul;   % liquid
global omega;  % wall

% solve for dmt=Z/N, keep in mind mt1=mt0+dmt

Z1=mt1-(rhol/mul)*(omega^2)*ri0*(sin(a)-cos(a)*(d1-d0)/dx)*(1/3)*(d0^3)*2*pi*ri0*rhol;
N1=1-(1/2)*((d0^2)/(mul*dx))*(uv*cos(a)+uld)*2*pi*ri0*rhol;
dmt=Z1/N1;

b=2;