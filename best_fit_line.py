from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

X = np.array([2,3,4,5,6,7,8])
Y = np.array([5,9,12,10,14,8,19])

def best_fit(X,Y):
    m = ((mean(X)*mean(Y))-(mean(X*Y)))/((mean(X)*mean(X))-(mean(X*X)))
    c = mean(Y)-(m*mean(X))
    return m,c

m, c = best_fit(X,Y)

reg_line = [(m*x)+c for x in X]

plt.scatter(X,Y)
plt.plot(X,reg_line)
plt.show()
