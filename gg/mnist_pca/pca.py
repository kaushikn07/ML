
from sklearn.datasets import load_digits
import pandas as pd
mnist = load_digits()
df2 = pd.DataFrame(mnist.data)
df = df2.iloc[:,:-1]
cp = df2.iloc[:,-1]
df.head()
import numpy as np
arr = df.to_numpy()
arr
arr_m = arr - np.mean(arr , axis = 0)
cov_mat = np.cov(arr_m , rowvar = False)
eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
sorted_index = np.argsort(eigen_values)[::-1]
sorted_eigenvalue = eigen_values[sorted_index]
sorted_eigenvectors = eigen_vectors[:,sorted_index]
n_components = 2
eigenvector_subset = sorted_eigenvectors[:,0:n_components]
arr_r = np.dot(eigenvector_subset.transpose(),arr_m.transpose()).transpose()
print(arr_r)
principal_df = pd.DataFrame(arr_r , columns = ['PC1','PC2'])
principal_df = pd.concat([principal_df , pd.DataFrame(cp)] , axis = 1)
import seaborn as sb
import matplotlib.pyplot as plt

plt.figure(figsize = (6,6))
sb.scatterplot(data = principal_df , x = 'PC1',y = 'PC2' , s = 60 , palette= 'icefire')

'''from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from flask import Flask, render_template

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
print(len(mnist))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

# Apply PCA to reduce the dimensionality of the data
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Initialize the Flask application
app = Flask(__name__)

# Define a route for computing the accuracy
@app.route('/')
def get_accuracy():
    clf = LogisticRegression()
    clf.fit(X_train_pca, y_train)
    accuracy = clf.score(X_test_pca, y_test)
    return render_template('accuracy.html', accuracy=accuracy)

# Start the Flask application
if __name__ == '__main__':
    app.run()'''
