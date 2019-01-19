import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
"""
high and low -> diff -> volutality
open and clode -> price go up or down
Simple regression will not be able to identify these relationships
So its better if we are able to extract these relationships first and use
regression on that instead of just numbers.
"""
df['high_low_perc_change'] = ((df['Adj. High'] - df['Adj. Low'])/df['Adj. High'])*100.0
df['close_open_perc_change'] = ((df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'])*100.0
df = df[['Adj. Close', 'high_low_perc_change', 'close_open_perc_change', 'Adj. Volume']]

#print(df.head())

forecast_col = 'Adj. Close'
#-99999 will just set it as an outlier, advantage-> we are not getting rid of the data
df.fillna(-99999, inplace=True)

#predict 10%
forecast_out = int(math.ceil(0.01*len(df)))

#so now we are shifting up by 30 days, so we will have a label col
#day for any row given we will have the label col with it closing price after 30 days
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
#print(df.head())

#{0 or ‘index’, 1 or ‘columns’}, default 0
x = np.array(df.drop(['label'],1))
y = np.array(df['label'])

"""
Source: Sklearn, https://scikit-learn.org/stable/modules/preprocessing.html
Standardization of datasets is a common requirement for many machine
learning estimators implemented in scikit-learn;
they might behave badly if the individual features do not more or less
look like standard normally distributed data: Gaussian with zero mean and unit variance.
In practice we often ignore the shape of the distribution and just transform the data to
center it by removing the mean value of each feature, then scale it by dividing
non-constant features by their standard deviation.

For instance, many elements used in the objective function of a learning
algorithm (such as the RBF kernel of Support Vector Machines or the l1 and
l2 regularizers of linear models) assume that all features are centered around
zero and have variance in the same order. If a feature has a variance that is
orders of magnitude larger than others,
it might dominate the objective function and make the estimator unable to learn
from other features correctly as expected.
"""
x = preprocessing.scale(x)
#print(len(x),len(y))


#splitting the data into 20% test data
x_train, y_train, x_test, y_test = cross_validation.train_test_split(x, y, test_size=0.2)
clf = LinearRegression()
#train
clf.fit(x_train, y_train)
#test
accuracy = clf.score(x_test, y_test)
print(accuracy)


















