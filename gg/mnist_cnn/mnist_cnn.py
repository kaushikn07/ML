import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Define the CNN model
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(x_train.reshape(-1,28,28,1), y_train, epochs=10, batch_size=32, validation_data=(x_test.reshape(-1,28,28,1), y_test))

# Evaluate the model on the test data
test_loss, test_acc = model.evaluate(x_test.reshape(-1,28,28,1), y_test)

print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

model.save('mnist_cnn.h5')
