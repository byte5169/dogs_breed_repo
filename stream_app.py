import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

path = Path()
learn_inf = load_learner(path/'dogs_breed_standford.pkl', cpu=True)
header = Image.open('logo.jpg')

st.image(header, use_column_width=True)

st.write("""
# A simple dog breed classifier!

""")

st.write("""
Model that predicts one of the 120 dog breeds. \n 
Data used for model training was taken from Kaggle's [Standford Dogs dataset](
https://www.kaggle.com/jwyang91/dog-breed-classification-using-fastai/)
***
""")
upload_img = st.file_uploader(type=['png', 'jpg', 'jpeg'], label='Upload here:')

def predict_dog(dog_image):
    img_t = tensor(dog_image)
    pred, pred_idx, probs = learn_inf.predict(img_t)
    pred = pred.split('-')[1].capitalize()
    probs = str(round((probs[pred_idx].item() * 100), 2)) + ' %'
    return pred, probs


if upload_img is not None:
    dog = Image.open(upload_img)
    the_dog = predict_dog(dog)
    st.write('Prediction:', the_dog[0])
    st.write('Probability:', the_dog[1])
    st.image(dog.to_thumb(128,128))






