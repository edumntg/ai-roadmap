clc, clear all, close all

g = 9.81;
p = 1000;
R = 7e-2;

kout = R*p*g;
tf = 10;
N = 10000;
dt = tf/N;
tspan = linspace(0, tf, N);

P = 50;

H = zeros(N,1);
H(1) = 1; % initial
for i = 1:N-1
    ti = (i-1)*dt;
    
    Q = kout*sqrt(H(i));
    
    H(i+1) = H(i) + (dt/2)*(P - Q);
    if H(i+1) < 0
        H(i+1) = 1e-6;
    end
end

figure
plot(tspan, H)

    