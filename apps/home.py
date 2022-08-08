import streamlit as st

def app():

    def header(url):
     st.markdown(f'<p style="color:white;font-size:48px;font-weight:bold">{url}</p>', unsafe_allow_html=True)

    header("Welcome to StockStream!")

    st.markdown(
    """
    <style>
    .stApp {
        
    background-image: url("https://img.etimg.com/thumb/width-1200,height-900,imgsize-299518,resizemode-1,msid-79611089/prime/money-and-markets/is-nifty-overvalued-a-short-term-correction-may-settle-the-debate-.jpg");
    background-size: 1050px;
    background-repeat: no-repeat;
    background-position: right;
    }
    </style>
    """,
    unsafe_allow_html=True
    )
