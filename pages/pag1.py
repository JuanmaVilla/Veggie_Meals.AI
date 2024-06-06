import streamlit as st
import time
import streamlit as st


st.title("Creando ideas")
st.subheader("Suba los ingredientes")

st.file = st.file_uploader("Pick a file")

import streamlit as st

st.toast('Your edited image was saved!', icon='🥙')

click = st.button("Crear receta")

with st.spinner('Wait for it...'):
    time.sleep(5)
st.success('Done!')

def cook_breakfast():
    msg = st.toast('Gathering ingredients...')
    time.sleep(1)
    msg.toast('Cooking...')
    time.sleep(1)
    msg.toast('Ready!', icon = "🥞")

if st.button('Cook breakfast'):
    cook_breakfast()

