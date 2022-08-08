import streamlit as st

def app():
    def header(url):
     st.markdown(f'<p style="color:black;font-size:48px;font-weight:bold;">{url}</p>', unsafe_allow_html=True)

    header("About")

    st.markdown(
    """
    <style>
    .stApp {
        
    background-image: url("https://www.pageuppeople.com/wp-content/uploads/2019/01/Top-60-Employee-Engagement-image43-1200x720.png");
    background-size: 600px;
    background-repeat: no-repeat;
    background-position: right;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    def lines(line):
        st.markdown(f'<p style="color:black;font-size:20px;">{line}</p>', unsafe_allow_html=True)

    lines('StockStream is your one stop')
    lines('destination for projecting and') 
    lines('visualising stock market data')
    lines('for 60 days into the future.')
    lines('We have made this app keeping in')
    lines('mind the priorities of the modern')
    lines('day investor i.e. time and efficiency!')