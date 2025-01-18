import numpy as np
import tensorflow as tf
from flask_cors import CORS
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify

# Initialize the Flask application
app = Flask(__name__)
# Enable CORS to allow cross-origin requests
CORS(app)

# Load the pre-trained TensorFlow model
# The "model" folder contains the saved TensorFlow model
model = tf.saved_model.load("C:\\Users\\KRoom\\Documents\\Owen_Project\\Intern_Project\\model")
# Retrieve the default inference function from the loaded model
infer = model.signatures["serving_default"]

# Print the structured input signature to understand the input format the model expects
print(infer.structured_input_signature)
# Print the structured output signature to understand the output format of the model
print(infer.structured_outputs)

# Define an API route for prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the JSON data sent in the POST request
        data = request.json
        # Extract the features (image data) and reshape them to the model's input shape
        # Expecting input shape (batch_size, 28, 28, 1) for grayscale images
        features = np.array(data["features"]).reshape(len(data["features"]), 28, 28, 1)

        plt.figure(figsize=(10, 5))
        for i in range(len(features)):
            plt.subplot(1, len(features), i + 1)
            plt.imshow(features[i], cmap="gray")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

        # Convert the NumPy array of features to a TensorFlow tensor
        # The model expects inputs as tensors of type float32
        inputs = tf.convert_to_tensor(features, dtype=tf.float32)

        # Perform inference using the loaded model
        # Pass the inputs to the model and retrieve predictions
        # "conv2d_input" is the expected input key for the model
        predictions = infer(conv2d_input=inputs)["dense_1"].numpy()
        # Convert the predictions (probabilities) to digit labels
        predicted_digits = [int(np.argmax(prediction)) for prediction in predictions]

        # Return the predictions as a JSON response
        return jsonify({"prediction" : predicted_digits})
    except Exception as e:
        # Handle any errors that occur during prediction
        # Return the error message as a JSON response
        return jsonify({"error" : str(e)})

# Run the Flask application in debug mode
if __name__=="__main__":
    app.run(debug=True)