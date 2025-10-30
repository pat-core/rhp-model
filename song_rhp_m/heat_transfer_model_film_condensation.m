function   [dmt, Twi, qw] = heat_transfer_model_film_condensation( d0,d1, Two, Tsat, dx,ri,ro, ~ )
%FILM_CONDENSATION
% solve liquid flow model for difference in mass flow (liquid) 
% evaluation at mid-point x=(x0+x1)/2
% ri/ro...inner/outer radius at xm
% Twi/Two...inner/outer wall temperature at xm

global kl hfg ;   % liquid
global kw;  % wall

% FVM evaluation at midpoint x=x0/2+x1/2
d=d0/2+d1/2;
ln=ri*log(ro/ri);

if (d>0)  % there is a liquid layer
    Twi=((kl/d)*Tsat+(kw/ln)*Two)/(kw/ln+kl/d);
else   % there is no liquid layer
    Twi=Tsat;
end

qw=kw*(Twi-Two)/ln;   % qw is always determined in contrast to ql 
dmt=qw*2*pi*ri*dx/hfg;






