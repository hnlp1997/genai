import streamlit as st

st.title("Bay Area Food Chatbot :hamburger:")

with st.form(key='my_form'):
  text_input = st.text_input(label='Write a query asking about food in the Bay Area:')
  submit_button = st.form_submit_button(label='Submit')

