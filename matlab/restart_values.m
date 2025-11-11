function K = restart_values(N)
%RESTART_VALUES 
% generate list of factors

K=zeros(N+1,1); % last value must be zero, but is not used for iterations
N2=floor(N/2);
N21=N2+1;

for k=1:N2
    K(2*k-1)=(N21-k)/N21;
    K(2*k)=N21/(N21-k);
end

% some drastic values last
% K(end-2)=1/100;
% K(end-1)=100;

end
