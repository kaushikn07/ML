from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow import keras

app = Flask(__name__)

# Load the CNN model
model = keras.models.load_model("mnist_cnn.h5")

# Define a function to preprocess the image
def preprocess_image(image):
    # Scale the pixel values to the range of 0 to 1
    image = image.astype("float32") / 255.0
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    # Reshape the image to the expected input shape of the model
    image = np.reshape(image, (1,28,28,1))
    return image

# Define a function to make a prediction with the model
def make_prediction(image):
    # Preprocess the image
    image = preprocess_image(image)
    # Make a prediction with the model
    prediction = model.predict(image)
    # Get the index of the predicted class
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get the uploaded image file from the request
        file = request.files["image"]
        # Read the image file as a NumPy array
        image = np.fromstring(file.read(), np.uint8)
        # Decode the image as grayscale
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        # Make a prediction with the model
        prediction = make_prediction(image)
        # Return the prediction as JSON
        return jsonify({"prediction": str(prediction)})
    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    
'''from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.models import load_model

model = load_model("model.h5")

app = Flask(__name__)

def preprocessing(image):
    image = load_img(image,target_size=(28,28),color_mode = 'greyscale')
    image = img_to_array(image)
    image = image.reshaper(1,28,28,1)
    image = image.astype('float32')/255.0
    return image

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template(home.html)

@app.route("/prediciton", methods = ["POST"])
def predictions():
    img = request.files['img']
    img = preprocessing(img)
    result = model.predict(img)
    return render_template(result.html,result = result)

app.run(debug = True)'''
