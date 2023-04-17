import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def neural_network(X, y, hidden_size, num_epochs, learning_rate):
    # Initialize weights
    input_size = X.shape[1]
    output_size = y.shape[1]
    W1 = np.random.randn(input_size, hidden_size)
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size)
    b2 = np.zeros((1, output_size))
    
    # Training loop
    for i in range(num_epochs):
        # Forward propagation
        z1 = np.dot(X, W1) + b1
        a1 = sigmoid(z1)
        z2 = np.dot(a1, W2) + b2
        y_pred = sigmoid(z2)
        
        # Backpropagation
        delta2 = (y_pred - y) * sigmoid_derivative(y_pred)
        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(a1)
        
        # Update weights
        dW2 = np.dot(a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        W2 -= learning_rate * dW2
        b2 -= learning_rate * db2
        W1 -= learning_rate * dW1
        b1 -= learning_rate * db1
    
    return W1, b1, W2, b2
