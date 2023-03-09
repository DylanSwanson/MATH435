clear all
load marketdata

%This program solves the maximin game with the investor as
%the column play.
%A = Y';%this makes Fate the row player following a minimax strategy
A = A(1:4,:)
%A = A(:,[1 4 5 7]) This plays the smaller game.

[m n] = size(A);
em = ones(m,1);
en = ones(n,1);
c = [0*en' 1]'
Aineq = [-A em];
Aeq = [en' 0];
bineq = 0*em;
beq = 1;

lb=[0*en' -Inf]';
%we need to take into account that the decision variables are 
%bounded below by zero but the value of the games is not (it is free)

[xx zz] = linprog(-c, Aineq, bineq, Aeq, beq, lb);
display('Strategy for Column Player')
xx
display('Payoff for either player')
%Be sure to negate the result since we are solving 
%max zeta which is the same as -min-zeta
-zz