% Loads necessary data.
load marketdata

% Calculations for column and row player strategies.
[colY, colZ] = columnPlayer(A);
[rowY, rowZ] = rowPlayer(A);

% Displaying values
display('Strategy for Column Player')
colY
display('Payoff for Column Player')
colZ
display('Strategy for Row Player')
rowY
display('Payoff for Row Player')
rowZ

% Function to calculate strategy for column player.
% @param(A) : The payoff matrix. This can be the same matrix given to rowPlayer()
% @return(y) : The decision vector for the column player.
% @return(z) : The payoff amount for the column player.
function [y, z] = columnPlayer(A)
    
    % m is the number of rows and n is the number of columns.
    [m n] = size(A);
    % Creating a row vector of m ones.
    % Note that em and en are row vectors, unlike the source code from Dr. Kirby.
    % This is done this way to more closely align with the myportfolio3 code.
    em = ones(1,m);
    % Creating a row vector of n ones.
    en = ones(1,n);

    % Creates a column vector of n zeroes followed by 1 one.
    % Note c is a (n+1 x 1) matrix.
    c = [0*en 1]';

    % A_inequality calculation from lecture.
    % Note A_inequality is a (m x n+1) matrix.
    A_in = [-A em'];

    % A_equality calculation from lecture.
    % Note A_equality is a (1 x n+1) matrix.
    A_eq = [en 0];

    % b_inequality calculation from lecture.
    % Note b_inequality is a (m x 1) matrix.
    b_in = 0*em';

    % b_equality calculation from lecture.
    b_eq = 1;

    % Lower bound calculation from lecture.
    % Note lb is a (n+1 x 1) matrix.
    lb = [0*en -Inf]';

    % Use of linprog from lecture.
    [y z] = linprog(-c, A_in, b_in, A_eq, b_eq, lb);

    % Note z is negated becuase we are maximizing for the column player but linprog minimizes.
    % This is also why c is negated in the linprog function call.
    z = -z;
end

% Function to calculate strategy for row player.
% @param(A) : The payoff matrix. This can be the same matrix given to columnPlayer()
% @return(y) : The decision vector for the row player.
% @return(z) : The payoff amount for the row player.
function [y, z] = rowPlayer(A)
    
    % m is the number of rows and n is the number of columns.
    [m n] = size(A);
    % Creating a row vector of m ones.
    % Note that em and en are row vectors, unlike the source code from Dr. Kirby.
    % This is done this way to more closely align with the myportfolio3 code.
    em = ones(1,m);
    % Creating a row vector of n ones.
    en = ones(1,n);

    % Creates a column vector of n zeroes followed by 1 one.
    % Note c is a (m+1 x 1) matrix.
    c = [0*em 1]';

    % A_inequality calculation from lecture.
    % Note A_inequality is a (n x m+1) matrix.
    A_in = [-A' en'];

    % A_equality calculation from lecture.
    % Note A_equality is a (1 x m+1) matrix.
    A_eq = [em 0];

    % b_inequality calculation from lecture.
    % Note b_inequality is a (n x 1) matrix.
    b_in = 0*en';

    % b_equality calculation from lecture.
    b_eq = 1;

    % Lower bound calculation from lecture.
    % Note lb is a (m+1 x 1) matrix.
    lb = [0*em -Inf]';

    % Use of linprog from lecture.
    % Note A_inequality is negated.
    [y z] = linprog(c, -A_in, b_in, A_eq, b_eq, lb);
end
