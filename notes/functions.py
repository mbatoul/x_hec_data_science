import numpy as np
import matplotlib.pyplot as plt
import math

X = np.array(range(-10, 10))
LAMBDA = 1
def func(k):
  return math.exp(-LAMBDA) * (LAMBDA ** k / math.factorial(k))
poisson = np.vectorize(func)
Y = poisson(X)
plt.plot(X, Y)
plt.show()