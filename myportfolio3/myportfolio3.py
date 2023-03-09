from scipy.optimize import linprog
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt

# Load original dataset.
# Rows represent individual years.
# Values represent returns on specific instrument (column)
# for specific year (row).
A = np.array([
    [1.103, 1.159, 1.061, 1.030, 0.903, 1.150, 1.074, 0.825],
    [1.080, 1.366, 1.316, 1.326, 1.333, 1.213, 1.562, 1.006],
    [1.063, 1.309, 1.186, 1.161, 1.086, 1.156, 1.694, 1.216],
    [1.061, 0.925, 1.052, 1.023, 0.959, 1.023, 1.246, 1.244],
    [1.071, 1.086, 1.165, 1.179, 1.165, 1.076, 1.283, 0.861],
    [1.087, 1.212, 1.316, 1.292, 1.204, 1.142, 1.105, 0.977],
    [1.080, 1.054, 0.968, 0.938, 0.830, 1.083, 0.766, 0.922],
    [1.057, 1.193, 1.304, 1.342, 1.594, 1.161, 1.121, 0.958],
    [1.036, 1.079, 1.076, 1.090, 1.174, 1.076, 0.878, 0.926],
    [1.031, 1.217, 1.100, 1.113, 1.162, 1.110, 1.326, 1.146],
])

# This is the following year from A.
# This is used for profit analysis.
X1994 = np.array([1.045, 0.889, 1.012, 0.999, 0.968, 0.965, 1.078, 0.990])

# Vector of mu values. [0, 10] in steps of 0.01.
mu_vector = np.linspace(0, 10, 1001)

# The column means for A
# Note this is a row vector.
f = np.mean(A, 0)

# m is the number of rows and n is the number of columns.
m, n = A.shape
# Creating a row vector of m ones.
em = np.ones(m)
# Creating a row vector of n ones.
en = np.ones(n)

# These varaiables are needed for plotting later.
XX = np.empty((n+m, len(mu_vector)))
reward = np.empty(len(mu_vector))
risk = np.empty(len(mu_vector))
payoff = np.empty(len(mu_vector))

# i represents the index within the vector, and mu is the specific mu
# value at that index.
for i, mu in enumerate(mu_vector) :

    # Calculate c_1 based on what was shown in lecture.
    # Note c_1 is a column vector to align more properly with lecture.
    c_1 = np.reshape((mu * f), (n, 1))

    # Calculate c_2 based on what was shown in lecture.
    # Note c_2 is a column vector to align more properly with lecture.
    c_2 = np.reshape((-em / m), (m, 1))

    # Combine c_1 and c_2 to get c.
    # Note c is a column vector to align more properly with lecture.
    c = np.vstack((c_1, c_2))

    # S is the matrix whose m rows are all equal to the above f (column 
    # means of A).
    S = repmat(f,m,1)

    # Calculate A_tilde (column mean subtracted from each column of payoff 
    # matrix).
    A_tilde = A-S

    # Creates an identity matrix of size (m x m).
    I = np.identity(m)

    # A_inequality calculation from lecture.
    # Note A_inequality is a (2m x n+m) matrix.
    A_in = np.vstack((np.hstack((-A_tilde, -I)),
                    np.hstack((A_tilde, -I))
    ))

    # b_inequality calculation from lecture.
    # Note b_inequality is a (2m x 1) row vector of all zeros.
    b_in = np.zeros((2*m, 1))

    # A_equality calculation from lecture.
    # Note A_equality is a (1 x n+m) row vector of n ones followed by m
    # zeros.
    A_eq = np.reshape(np.hstack((en, 0*em)), (1, n+m)) 

    # b_equality calculation from lecture.
    b_eq = 1

    # Use of linprog from lecture.
    # Note result is a scipy.optimize.OptimizeResult object
    # Note bounds are not specified as the default is already all non-negative
    result = linprog(-c, A_in, b_in, A_eq, b_eq)

    # Select the y decision vector for easier use later.
    # Note y is now a row vector of shape (1 x n).
    d = result.x
    y = d[:n]

    # The reward for this particular value of mu is the decision vector, y,
    # times the column means, f.
    # Note since f is a row vector, it must be transposed to get a single 
    # value answer.
    reward[i] = y @ np.transpose(f)

    # Select the z vector for easier use later.
    # Note z is now a row vector of shape (1 x m).
    z = d[n+1:]

    # The risk for this particular value of mu is the mean of the z values.
    risk[i] = np.mean(z)

    # XX is used for storing the decision vectors for later plotting.
    XX[:, i] = np.transpose(d)

    # Payoff is a row vector containing the payoff amounts of the
    # previously calculated decision vector against the 1994 data.
    payoff[i] = y @ np.transpose(X1994)

# Adjust this to fit the plot size desired
plotSize = 10

# Plots the reward versus risk graph. Figure 1.
fig1 = plt.figure(1, figsize=(plotSize, plotSize))
plt.plot(reward, risk, '--o')
plt.xlabel('reward')
plt.ylabel('risk')
plt.title('Figure 1')

# Plots the decision vectors with repsect to the mu value. Figure 2.
fig2 = plt.figure(2, figsize=(plotSize, plotSize))
labels = ['Bonds', 'Materials','Energy','Financial','Industrial','Technology','Staples','Utilities']
for i, label in enumerate(labels) :
    plt.plot(mu_vector, XX[i, :], label=label)
plt.xlabel('risk parameter')
plt.ylabel('Strategy')
plt.title('Figure 2')
plt.legend()

# Plots the payoff versus mu graph. Figure 3.
fig3 = plt.figure(3, figsize=(plotSize, plotSize))
plt.plot(mu_vector, payoff)
plt.xlabel('Risk parameter')
plt.ylabel('Payoff for year 1994')
plt.title('Figure 3')

plt.show()