import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

## get the training data
## first element is 1 because of the bias term, then x1, x2 follows

X = np.array([
    [1, 1, 4],
    [1, 1, 1],
    [1, 2, 2],
    [1, 3, 1],
    [1, 5, 4],
    [1, 5, 5],
    [1, 4, 4],
    [1, 2, 5]]
)

# labels for the training data
Y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

## function to get the derivative, rc = lambda
def get_derivative(X, Y, w, rc):
    der = np.zeros(len(X[0]))
    
    total = 0
    correct = 0
    
    ## for all the training samples
    for x, y in zip(X, Y):
        if y * np.dot(x, w) < 1:
            der = der - y * x
        der = der + (2.0 / rc) * w
    
    return der

iterations = 120000
w = np.zeros(len(X[0]))
rate = 1.25
rc = 11000

## do the training
for i in xrange(iterations):
    w = w - rate * get_derivative(X, Y, w, rc)

print w

## plot the results
for i in xrange(len(X)):
    if Y[i] == 1:
        mark = "+"
    else:
        mark = "_"
    plt.scatter(X[i][1], X[i][2], marker = mark, s = 150)
    
plt.plot([0, -w[0]/w[2]], [-w[0]/w[1],0])
