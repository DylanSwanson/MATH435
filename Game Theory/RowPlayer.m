
load marketdata

%Hi John.  Let's consider the matrix game of matching pennies.

%first clear all the variables in the workspace.
%In penny matching the payoff matrix looks like
%A = [0 1 -2; -3 0 4; 5 -6 0];
%look at the notes for a discussion.  Basically if (column) 
%player C selects col 1 and row player R selects row 1 the payoff 
%is -1 (goes to player R) and so on.

%if I wanted to save this data in a matlab mat file I would use
%the command
%save penniesdata A

%to load that data back I would use the command
%load penniesdata

%now, we need to set the matrix up according to equation 
%11.4 in the text (this is the row player's minimax strategy).

%We need the size of the matrix.  In matlab you can do this as
[m n] = size(A);
%now m is the number of rows and number of cols.

%We need column vectors of m ones (1 .... 1)'
em = ones(m,1);
%and of n ones
en = ones(n,1);

%We dot the decision var with [0 1] but note that the 0 is a vec
b = [0*em' 1]'
%we pass the inequality constraints separately from the equality
Aineq = [-A' en];
% the one row of equality constraints
Aeq = [em' 0];
bineq = 0*en;
beq = 1;

%We need to specify that the v variable is free
lb=[0*em' -Inf]'

%the call to the linear program
[y e] = linprog(b, -Aineq, bineq, Aeq, beq, lb)

%y contains the values of the decision variables
%e is the value of the objective function