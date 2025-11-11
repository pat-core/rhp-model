function   [dmt, Twi, qw] = heat_transfer_model_mixed_convection( d0,d1, Two, Tsat, dx,ri,ro, a )
%MIXED_CONVECTION 
% solve liquid flow model for difference in mass flow (liquid) 
% used in evaporator, evaluation at x=xm
% ri/ro...inner/outer radius at xm
% Twi/Two...inner/outer wall temperature at xm

global kl hfg Pr;   % liquid
global kw;  % wall

Nuf=1;  % Nusselt Number of forced convection
d=d0/2+d1/2;
ln=ri*log(ro/ri);

%% first run with Twi=Two
Gr=grashof(ri, a, Tsat-Two, d);  % here use Two
Ra=Gr*Pr;
Nun=0.133*Ra^0.375;     % Nusselt Number of natural convection..
Num=(Nuf^(7/2)+Nun^(7/2))^(2/7);    % ..and mixed 
if (d>0)  % there is a liquid layer
    Twi=((Num*kl/d)*Tsat+(kw/ln)*Two)/(kw/ln+Num*kl/d);
else   % there is no liquid layer
    Twi=Tsat;
end

%% second run with Twi=Twi_first run
Gr=grashof(ri, a, Tsat-Twi, d);  % here use Twi 
Ra=Gr*Pr;
Nun=0.133*Ra^0.375;     % Nusselt Number of natural convection..
Num=(Nuf^(7/2)+Nun^(7/2))^(2/7);    % ..and mixed 
if (d>0)  % there is a liquid layer
    Twi=((Num*kl/d)*Tsat+(kw/ln)*Two)/(kw/ln+Num*kl/d);
else   % there is no liquid layer
    Twi=Tsat;
end

%% heat flow and mass flow difference

qw=kw*(Twi-Two)/ln;   % qw is always determined in contrast to ql 
dmt=qw*2*pi*ri*dx/hfg;

