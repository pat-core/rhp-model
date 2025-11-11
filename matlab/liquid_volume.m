function V = liquid_volume(d,RI)
%LIQUID_VOLUME
% d...film heigth in condenser (nodal)
% RI...inner radius (mid-point)
global DX;

deltamid=d(1:end-1)/2 + d(2:end)/2;
V= 2*pi*sum( deltamid.*DX.*RI);

end

