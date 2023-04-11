from flask import Flask, render_template
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import io

# Load the Iris dataset and create a decision tree classifier object
iris = load_iris()
clf = DecisionTreeClassifier(max_depth=4, min_samples_split=4, min_samples_leaf=3)
clf.fit(iris.data, iris.target)

# Export the decision tree to a dot file
dot_data = export_graphviz(clf, out_file=None, feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True, special_characters=True)

# Create a Pydot graph object from the dot data
graph = pydotplus.graph_from_dot_data(dot_data)

# Define a Flask app
app = Flask(__name__)

# Define a route to display the decision tree
@app.route('/')
def decision_tree():
    # Save the decision tree as a PNG image in memory
    png = io.BytesIO()
    graph.write_png(png)
    png.seek(0)

    # Return the image as a response to the request
    return render_template('decision_tree.html', image=png.getvalue().decode())

if __name__ == '__main__':
    app.run(debug=True)
    
'''from sklearn.model_selection import GridSearchCV

# Define the hyperparameters to test
params = {'max_depth': [2, 4, 6, 8, 10], 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4, 5]}

# Create a decision tree classifier object
clf = DecisionTreeClassifier()

# Create a grid search object
grid_search = GridSearchCV(clf, param_grid=params, cv=5)

# Fit the grid search object to the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and the corresponding accuracy score
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)'''
