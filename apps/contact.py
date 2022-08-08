import streamlit as st

def app():
    def header(url):
     st.markdown(f'<p style="color:black;font-size:48px;font-weight:bold">{url}</p>', unsafe_allow_html=True)

    header("Contact Us")

    def info(inf):
        st.markdown(f'<p style="color:black;font-size:20px;font-weight:bold">{inf}</p>', unsafe_allow_html=True)

    info("Email: help@stockstream.com")
    info("Phone: (485)555-1212")

    st.markdown(
    """
    <style>
    .stApp {
        
    background-image: url("https://www.stockspots.eu/wp-content/uploads/2019/10/contact-1.png");
    background-size: 700px;
    background-repeat: no-repeat;
    background-position: right;
    }
    </style>
    """,
    unsafe_allow_html=True
    )