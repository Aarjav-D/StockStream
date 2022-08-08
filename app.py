import pandas as pd
from multiapp import MultiApp
import streamlit as st
from apps import home, predictor, about, contact

app = MultiApp()

st.set_page_config(page_title="StockStream", page_icon="https://cdn0.iconfinder.com/data/icons/apple-apps/100/Apple_Stock-512.png")

st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">', unsafe_allow_html=True)

st.markdown(""" <style>
footer {visibility: hidden;}
</style> """, unsafe_allow_html=True)

st.markdown("""
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #3498DB; height:60px;">
  <img src="https://cdn0.iconfinder.com/data/icons/apple-apps/100/Apple_Stock-512.png" style="height: 60px;"/>
  <h2 style="color:orange; margin-top: 10px;">StockStream</h2>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  
</nav>
""", unsafe_allow_html=True)

app.add_app("home", home.app)
app.add_app("predictor", predictor.app)
app.add_app("About", about.app)
app.add_app("Contact Us", contact.app)

