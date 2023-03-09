from scipy.optimize import linprog
from scipy.io import loadmat
import numpy as np

def columnPlayer(A) :
    """ Function to calculate strategy for column player.
    @param(A) : The payoff matrix. This can be the same matrix given to rowPlayer()
    @return(y) : The decision vector for the column player.
    @return(z) : The payoff amount for the column player. """

    # m is the number of rows and n is the number of columns.
    m, n = A.shape

    # Creating a row vector of m ones.
    # Note that em and en are row vectors, unlike the source code from Dr. Kirby.
    # This is done this way to more closely align with the myportfolio3 code.
    em = np.ones(m)
    # Creating a row vector of n ones.
    en = np.ones(n)

    # Creates a column vector of n zeroes followed by 1 one.
    # Note c is a (n+1 x 1) matrix.
    c = np.reshape(np.hstack((0*en, np.array([1]))), (n+1, 1))

    # A_inequality calculation from lecture.
    # Note A_inequality is a (m x n+1) matrix.
    A_in = np.hstack((-A, np.reshape(em, (m, 1))))

    # A_equality calculation from lecture.
    # Note A_equality is a (1 x n+1) matrix.
    A_eq = np.reshape(np.hstack((en, np.array([0]))), (1, n+1))

    # b_inequality calculation from lecture.
    # Note b_inequality is a (m x 1) matrix.
    b_in = np.reshape(0*em, (m, 1))

    # b_equality calculation from lecture.
    b_eq = 1

    # Lower bound calculation from lecture.
    # Sets all bounds to be positive.
    lb = [(0, None)] * (n+1)
    # Sets last bound to be unrestricted.
    lb[n] = (None, None)
    # Note lb is a (n+1 x 1) matrix.
    lb = np.array(lb)

    # Use of linprog from lecture.
    # Note result is a scipy.optimize.OptimizeResult object
    result = linprog(-c, A_in, b_in, A_eq, b_eq, lb)

    # Select the decision vector from result
    y = result.x

    # Select the optimal value from result
    z = result.fun

    # Note z is negated becuase we are maximizing for the column player but linprog minimizes.
    # This is also why c is negated in the linprog function call.
    return y, -z

def rowPlayer(A) :
    """ Function to calculate strategy for row player.
    @param(A) : The payoff matrix. This can be the same matrix given to columnPlayer()
    @return(y) : The decision vector for the row player.
    @return(z) : The payoff amount for the row player. """

    # m is the number of rows and n is the number of columns.
    m, n = A.shape

    # Creating a row vector of m ones.
    # Note that em and en are row vectors, unlike the source code from Dr. Kirby.
    # This is done this way to more closely align with the myportfolio3 code.
    em = np.ones(m)
    # Creating a row vector of n ones.
    en = np.ones(n)

    # Creates a column vector of n zeroes followed by 1 one.
    # Note c is a (m+1 x 1) matrix.
    c = np.reshape(np.hstack((0*em, np.array([1]))), (m+1, 1))

    # A_inequality calculation from lecture.
    # Note A_inequality is a (n x m+1) matrix.
    A_in = np.hstack((np.transpose(-A), np.reshape(en, (n, 1))))

    # A_equality calculation from lecture.
    # Note A_equality is a (1 x m+1) matrix.
    A_eq = np.reshape(np.hstack((em, np.array([0]))), (1, m+1))

    # b_inequality calculation from lecture.
    # Note b_inequality is a (n x 1) matrix.
    b_in = np.reshape(0*en, (n, 1))

    # b_equality calculation from lecture.
    b_eq = 1

    # Lower bound calculation from lecture.
    # Sets all bounds to be positive.
    lb = [(0, None)] * (m+1)
    # Sets last bound to be unrestricted.
    lb[m] = (None, None)
    # Note lb is a (m+1 x 1) matrix.
    lb = np.array(lb)

    # Use of linprog from lecture.
    # Note result is a scipy.optimize.OptimizeResult object
    result = linprog(c, -A_in, b_in, A_eq, b_eq, lb)

    # Select the decision vector from result
    y = result.x

    # Select the optimal value from result
    z = result.fun

    return y, z

# Loads necessary data
A = loadmat('./marketdata.mat').get('A')

# Calculations for column and player strategies.
colY, colZ = columnPlayer(A)
rowY, rowZ = rowPlayer(A)

# Prints values
print(f"Strategy for Column Player\n{colY}\nPayoff for Column Player"
      + f"\n{colZ}\nStrategy for Row Player\n{rowY}\nPayoff for Row Player\n{rowZ}")