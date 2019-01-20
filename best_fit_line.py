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

def sq_error(Y_orig, Y_new):
    return sum((Y_orig - Y_new )**2)

def r_sq(Y_orig, Y_new):
    Y_mean = [mean(Y_orig) for _ in Y_orig]
    sq_y_line = sq_error(Y_orig, Y_new)
    sq_y_mean = sq_error(Y_orig, Y_mean)
    return 1 - (sq_y_line/sq_y_mean)

m, c = best_fit(X,Y)

reg_line = [(m*x)+c for x in X]
print(r_sq(Y, reg_line))
plt.scatter(X,Y)
plt.plot(X,reg_line)
plt.show()
