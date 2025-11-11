function Re = reynolds(d,U)
%REYNOLDS 
% Reynolds Number of liquid flow
% d(elta)...film thickness
% U...average liquid velocity

global nul;

  Re=abs( 4*U*d/nul );
end

