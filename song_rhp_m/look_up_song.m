function ml = look_up_song(d2D)
%LOOK_UP_SONG 
% liquid mass as function of film height at evaporator end cap
% [Song 2003]

d2D_data=[0, 0.011, 0.019, 0.025, 0.036];
ml0=0.7e-3;
ml_data=[1, 3, 5, 7, 10]*ml0;

ml=interp1(d2D_data, ml_data, d2D);

end

