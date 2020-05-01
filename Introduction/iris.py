from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import mglearn
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

iris_dataset = load_iris()

# print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# print("\n")

# these are just the keys put in seperate vars
DESCR = iris_dataset['DESCR']
data = iris_dataset['data']
target = iris_dataset['target']
target_names = iris_dataset['target_names']
feature_names = iris_dataset['feature_names']
filename = iris_dataset['filename']

# shufffling the dataset before splitting it into 'training' and 'test' sets
	# to make sure that whenever I run this code I get the same result we use 'random_state=0'
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# ---------------------
# | LOOK AT YOUR DATA |
# ---------------------
# create dataframe (pandas datastructure) from data in X_train
# labeling columns using the strings from the iris_dataset.feature_names
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

# create a scatter matrix from the dataframe, color by y_train
# grr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)
# plt.show()

# -----------------------
# | k-Nearest Neighbors |
# -----------------------
# we specify how manyneighbors we are interested in -> what k is (in our case k = 1)
knn = KNeighborsClassifier(n_neighbors=1)
# SUPERVISED LEARNING
# now we classify the classes into categories
knn.fit(X_train, y_train)


# ----------------------
# | Making Predictions |
# ----------------------
# we can now make predictions based on our fit and kNN with k=1
X_new = np.array([[5,2.9,1,0.2]])
prediction = knn.predict(X_new)
print("prediction: {}".format(prediction))
print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))

# ------------------------
# | evaluating the model |
# ------------------------
# we will predict the classes of the test set and compare it to their actual values
y_pred = knn.predict(X_test)
print("Test set predictions:\n {}".format(y_pred))

print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))



