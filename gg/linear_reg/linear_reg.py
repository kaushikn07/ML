from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load the Boston Housing dataset
boston = load_boston()

'''# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(boston.data, boston.target, test_size=0.2, random_state=42)

# Create a linear regression model
lr_model = LinearRegression()

# Fit the model on the training data
lr_model.fit(X_train, y_train)

# Predict on the test data
y_pred = lr_model.predict(X_test)

# Calculate the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)


plt.scatter(boston.data[:, 5], boston.target, color='blue')
plt.plot(boston.data[:, 5], lr_model.predict(boston.data), color='red', linewidth=1)
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()'''

'''
boston = pd.DataFrame(data.data, columns=data.feature_names)
boston.head()
boston['MEDV'] = data.target
X = boston['AGE']
Y = boston['MEDV']
min(X),min(Y),max(X),max(Y)
def gradient_descent(m_cur,b_cur,data,L):
    
    m_grad = 0
    b_grad = 0
    
    n = len(data.data)
    
    for i in range(n):
        
        x = X[i]
        y = Y[i]
        
        
#         print(x,y)
        
        m_grad += -(2/n) * x * (y - (x * m_cur + b_cur))
        b_grad += -(2/n) * (y - x * m_cur + b_cur)
        
#         print(m_grad,b_grad)

    m = m_cur - m_grad * L
    b = b_cur - b_grad * L
    
    return m, b
L = 0.00001
epochs = 10000
m = 0
b = 0

for i in range(epochs):
    if(i % 50 == 0):
        print("epochs:", i)
    m, b = gradient_descent(m,b,data,L)

print(m,b)

from matplotlib import pyplot as plt

plt.scatter(X,Y)
plt.plot(list(range(3,100)),[x * m + b for x in list(range(3,100))])