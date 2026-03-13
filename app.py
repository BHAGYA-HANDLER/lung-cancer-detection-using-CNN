import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

IMG_SIZE = 128

classes = [
    "Benign",
    "Malignant",
    "Normal"
]

model = tf.keras.models.load_model("models/lung_cancer_model.h5")

st.title("Lung Cancer Detection using CNN")

st.write("Upload a CT scan image to detect lung cancer.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    result = classes[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(result)

# for running the frontend
# python -m streamlit run app.py