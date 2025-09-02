import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io

# Load model and class indices
def load_classifier():
    model = load_model('bacteria_model.h5')
    class_indices = np.load('class_indices.npy', allow_pickle=True).item()
    idx_to_class = {v: k for k, v in class_indices.items()}
    return model, idx_to_class

model, idx_to_class = load_classifier()

st.title('Bacteria Image Classifier')
st.write('Upload an image of a bacteria to detect its type.')

uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img = image.resize((64, 64))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred, axis=1)[0]
    class_name = idx_to_class[class_idx]
    st.success(f'Prediction: {class_name}')
