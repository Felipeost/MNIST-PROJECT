#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import cv2
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import joblib
from PIL import Image


# In[34]:


model = joblib.load("C:\\Users\\filip\\Documents\\EC\\Machine Learning\\Kunskapskontroll\\model.full")
scaler = joblib.load("C:\\Users\\filip\\Documents\\EC\\Machine Learning\\Kunskapskontroll\\scaler")

def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
 
    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(gray_image, (28, 28))
 
    # Normalize pixel values to the range [0, 1]
    normalized_image = resized_image / 255.0
 
    # Invert the image
    inverted_image = (1 - normalized_image) * 255
    
    # Delete pixels that are not dark enough
    mean_pixel_value = np.mean(inverted_image)
    inverted_image[inverted_image <= mean_pixel_value * 1.5] = 0
    
    return inverted_image

st.title('MNIST Digit Recognizer')

pages_names = ["Welcome Page", "Draw your own image", "Upload an image"]
pages = st.radio("Navigation", pages_names)

if pages == "Welcome Page":
    st.markdown('''
    This is a machine learning project created by Filip Östlund. The prediction model is created using the scikit-learn library.
    As you can see you can choose between drawing an image or uploading your own image. The model is not 100% correct but will do it's best ;)
    ''')

if pages == "Draw your own image": 
    st.subheader("Try to write a digit!")
    
    SIZE = 192
    canvas_result = st_canvas(
        fill_color='#000000',
        stroke_width=10,
        stroke_color='#FFFFFF',
        background_color='#000000',
        width=SIZE,
        height=SIZE,
        key='canvas')
    
    if canvas_result.image_data is not None:
        img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
        rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
        st.write('Model Input')
        st.image(rescaled)

    if st.button('Predict'):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flattened_image = gray_image.flatten()
        processed_image = flattened_image.reshape(1, -1)
        prediction = model.predict(processed_image)
        digit = prediction[0]
        st.write("Predicted Digit:", digit)

if pages == "Upload an image":
    # The message and nested widget will remain on the page
    st.subheader("Upload an image!")
    st.write('''For the prediction to work use a picture of a digit with a white background, preferably as little shadows and background "noise" as possible''')
    
    # Upload image through Streamlit
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", width = 300)
    
        # Preprocess the uploaded image
        preprocessed_image = preprocess_image(image)
    
        # Display the preprocessed image
        st.image((preprocessed_image / 255) , caption="Preprocessed Image.", width = 300)
    
        # Flatten to a 1D array
        processed_image = preprocessed_image.flatten()
    
        # Reshape the image array to match the input shape expected by the SVM model
        processed_image = processed_image.reshape(1, -1)
    
        # Scaling for model
        processed_image_scaled = scaler.transform(processed_image)
        
        # Make prediction
        prediction = model.predict(processed_image_scaled)
        digit = prediction[0]

        # Display the prediction
        st.write("Predicted Digit:", digit)


# In[ ]:




