import streamlit as st
from PIL import Image
import numpy as np
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
import os

# Title and header
st.title("🌼 Flower Classification App")


# Function to load and set up the model
@st.cache_resource
def load_model():
    vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in vgg_conv.layers[:-4]:
        layer.trainable = False
    model = Sequential([
        vgg_conv,
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(17, activation='softmax')
    ])
    # Load pre-trained weights if available
    weight_path = os.path.join("ModelWeights", "weights2.hdf5")
    if os.path.exists(weight_path):
        model.load_weights(weight_path)
    else:
        st.error("Model weights file not found. Please ensure 'weights2.hdf5' is in the 'ModelWeights' directory.")
    return model

# Mapping of class labels to flower names
flowers_dict = {
    'BlueShell': 0, 'Buttercup': 1, 'ColtsFoot': 2, 'Cowslip': 3, 'Crocus': 4,
    'Daffodil': 5, 'Daisy': 6, 'Dandelion': 7, 'Fritillary': 8, 'LilyValley': 9,
    'Pansy': 10, 'Snowdrop': 11, 'Sunflower': 12, 'TigerLily': 13, 'Tulip': 14,
    'WindFlower': 15, 'Iris': 16
}

# Reverse dictionary to find flower name from index
index_to_flower = {v: k for k, v in flowers_dict.items()}

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    # Display uploaded image
    uploaded_image = Image.open(uploaded_file)
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    image = np.array(uploaded_image.resize((224, 224)))
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    # Load the model and make predictions
    model = load_model()
    if model:
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction[0])
        flower_name = index_to_flower.get(predicted_class, "Unknown")

        # Display the prediction result
        st.success(f"Predicted label for the image is: {flower_name}")
else:
    st.info("Please upload an image of a flower for classification.")
