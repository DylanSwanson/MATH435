%load portfoliofull
A = [1.103 1.159 1.061 1.030 0.903 1.150 1.074 0.825;
1.080 1.366 1.316 1.326 1.333 1.213 1.562 1.006;
1.063 1.309 1.186 1.161 1.086 1.156 1.694 1.216;
1.061 0.925 1.052 1.023 0.959 1.023 1.246 1.244;
1.071 1.086 1.165 1.179 1.165 1.076 1.283 0.861;
1.087 1.212 1.316 1.292 1.204 1.142 1.105 0.977;
1.080 1.054 0.968 0.938 0.830 1.083 0.766 0.922;
1.057 1.193 1.304 1.342 1.594 1.161 1.121 0.958;
1.036 1.079 1.076 1.090 1.174 1.076 0.878 0.926;
1.031 1.217 1.100 1.113 1.162 1.110 1.326 1.146]

X1994 = [1.045 0.889 1.012 0.999 0.968 0.965 1.078 0.990]




muv = [.0:.01:10];
%muv = [6]
XX = [];

f = mean(A);%column means

[m n] = size(A)
em = ones(1,m);
en = ones(1,n);

for i = 1:size(muv,2)
    i

    c = [muv(i)*f/n em*(-1/m)]

    S = repmat(f,m,1);

    AA = A-S;%mean subtracted payoff matrix
    I1 = eye(m);

    Ain = [-AA -I1; AA -I1];
    bin = [zeros(2*m,1)];
    Aeq = [en 0*em] 
    beq = 1;

    [d val] = linprog(-c, Ain, bin, Aeq, beq, zeros(m+n,1))

    reward(i) = d(1:n)'*f';
    risk(i) = mean(d(n+1:n+m));
    XX = [XX d];
end

figure
plot(reward,risk,'--o')
xlabel('risk')
ylabel('reward')
figure
v=muv
plot(v,XX(1,:)',v,XX(2,:)',v,XX(3,:)',v,XX(4,:)',v,XX(5,:)',v,XX(6,:)',v,XX(7,:)',v,XX(8,:)','--y',v,XX(9,:)','-.')
legend('Bonds', 'Materials','Energy','Financial','Industrial','Technology','Staples','Utilities','Health')
xlabel('risk parameter')
ylabel('Strategy')
%legend('Bonds', 'Materials','Energy','Financial','Industrial','Technology','Staples','Utilities','Health')