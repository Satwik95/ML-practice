import numpy as np
from sklearn import preprocessing, cross_validation, neighbors, svm
import pandas as pd

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
print(df.describe())
df.replace('?',-99999, inplace=True)
"""
id is in most cases has no significance, since it lets the algo memorize the data,
hence in training it will perform really well for that particular data set
making it a high variance model, however not so well in the test data set.
The reason for this would be that the id has no qualitative relation with
the other features, it is related to only itself.

Thanks to explanation by: Khoi Hoang, youtube comment,
https://www.youtube.com/watch?v=mA5nwGoRAOo&index=20&list=PLQVvvaa0QuDfKTOs3Keq_kaG2P55YRn5v
"""
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = svm.SVC()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

#example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
#example_measures = example_measures.reshape(len(example_measures), -1)
#prediction = clf.predict(example_measures)
#print(prediction)
