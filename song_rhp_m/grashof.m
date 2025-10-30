function Gr = grashof(rim, a, dTm, deltam)
%GRASHOF
% Grashof Number at mid-point of FV
% rim...inner radius at mid-point of FV
% a...taper angle
% dTm...temperature difference (Tsat-Twi)
% deltam...film thickness at mid-point
global omega betal nul;

    Gr=abs( (omega^2)*rim*cos(a)*betal*dTm*(deltam^3)/(nul^2) );
    
end

