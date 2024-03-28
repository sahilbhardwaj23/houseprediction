import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Gurgaon Real Estate Analytics App",
    page_icon="🏠",
)

# Main title
st.title("Welcome to Real Estate Analysis & Prediction.")

# Introduction
st.subheader('This project is your secret weapon in the real estate game! Get ready to uncover hidden gems, '
              'predict property prices like a pro, and discover your dream home in Gurugram—all at your fingertips.')

# Navigation section title
st.title('Navigation')

# Predict Price section
st.header('Predict Price')
st.write("This section allows you to predict the price of houses and flats based on various features such as location, size, amenities, etc.")

# Analysis section
st.header('Analysis')
st.write("Dive deep into the real estate market in Gurugram by analyzing different aspects such as price trends, property types, location preferences, and more.")

# Recommendations section
st.header('Recommendations')
st.write("Get personalized recommendations for properties that match your preferences, making it easier for you to find your ideal home.")

# About section title
st.title('About')

# About section description
st.write("""
    This web application is designed to empower users to interactively explore and analyze real estate data in Gurugram. 
    Whether you're a potential buyer, seller, or simply curious about the real estate market, this platform provides valuable insights and tools to support your decision-making process.
""")

# Sidebar message
st.sidebar.success("Select a demo above.")
