import streamlit as st
from fastai.vision.all import *
from fastai.vision.widgets import *
from PIL import Image
from io import BytesIO

st.set_page_config(layout="centered")
path = Path()
learn_inf = load_learner('models/dogs_breed_standford_V2_20_01.pkl', cpu=True)
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

col1, col2 = st.beta_columns(2)
upload_img = col1.file_uploader(type=['png', 'jpg', 'jpeg'], label='Upload here to predict from image')
url_img = col2.text_input(label='or enter url here:')
but = col2.button(label='Press to predict from url')

def predict_dog(dog_image):
    img_t = tensor(dog_image)
    pred, pred_idx, probs = learn_inf.predict(img_t)
    pred = pred.split('-')[1].capitalize()
    probs = str(round((probs[pred_idx].item() * 100), 2)) + ' %'
    return pred, probs

def render_result(source):
    the_dog = predict_dog(source)
    st.text(f'Prediction: {the_dog[0]}')
    st.text(f'Probability: {the_dog[1]}')
    st.image(source.to_thumb(128, 128))


def fetch_url_image(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

if upload_img is not None:
    render_result(Image.open(upload_img))

try:
    if but:
        img_to_pred = fetch_url_image(url_img)
        predict_dog(img_to_pred)
        render_result(img_to_pred)
except:
    st.write("Unexpected error:", st.error())
