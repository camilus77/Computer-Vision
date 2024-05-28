
import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from contextlib import contextmanager
from time import time
from tqdm import tqdm

from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
import cv2


st.title('IMAGE CONVERSION SOFTWARE USING CYCLE GAN AND RESNET')
st.write('By Ubong C. Ben and Ntak Joshua')
st.write('This software takes and image and converts it to another image. \n')


st.header('INSTRUCTIONS')
st.write('1. Select original image\n2. Choose target Image\nClick ENTER to perform conversion')

#creat dropdown that selects the the image to be converted
st.header('INPUT')
inputt = st.selectbox(
    'What image do you wish to convert from?',
    ('-','Horse', 'Zebra', 'Van Gogh', 'Monet'))

lis={'Horse':'Zebra', 'Zebra':'Horse','Monet': 'Van Gogh','Van Gogh': 'Monet'}
if inputt=='Horse':
    st.write('You wish to convert from HORSE image to that of a ZEBRA')
elif inputt=='Zebra':
    st.write('You wish to convert from ZEBRA to HORSE')
elif inputt=='Monet':
    st.write('You wish to convert from Monet painting to Van Gogh painting')
elif inputt=='Van Gogh':
    st.write('You wish to convert from Van Gogh painting to Monet painting')
#create a text box to get the image number
ind = st.text_input('Input Image Here:') 

with st.form(key='form1'):
    submit=st.form_submit_button('Click here to view Input image')
    if submit:
        imagee = cv2.imread(f'images//{inputt} {ind}.jpg')
        cv2.imshow('Image', imagee)
        st.image(imagee, caption=f'INPUT IMAGE:{inputt}')
#get output out
st.header('OUTPUT')
with st.form(key='form2'):
    submit2=st.form_submit_button('Click here to make Image-to-Image conversion')
    if submit2:
        imagee = cv2.imread(f'images//{lis[inputt]} {ind}.jpg')
        cv2.imshow('Image', imagee)
        st.image(imagee, caption=f'OUTPUT IMAGE: {lis[inputt]}')
