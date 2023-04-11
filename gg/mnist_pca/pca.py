from sklearn.datasets import fetch_openml
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
    app.run()
