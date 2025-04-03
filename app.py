import streamlit as st
import pandas as pd
from datetime import datetime
from viewcount_prediction import *

def predict_top_10(date, country):
    movies_predictions, tv_predictions = main()
    return pd.DataFrame(movies_predictions), pd.DataFrame(tv_predictions)

# Streamlit App
st.title("Netflix Top 10 Rank Predictor")

# User Inputs
date = st.date_input("Select Date", datetime.today())
country = st.text_input("Enter Country", "United States")

if st.button("Predict Top 10"):
    predictions_movie, predictions_tv = predict_top_10(date, country)
    st.write("### Predicted Top 10 Shows/Movies")
    st.dataframe(predictions_movie)
    st.dataframe(predictions_tv)

    # Display DataFrame
    st.dataframe(predictions_movie)

    # Visualization
    st.write("### Predicted Ranks")
    fig, ax = plt.subplots()
    ax.barh(predictions_movie["show_title"], predictions_movie["predicted_hours"], color="skyblue")
    ax.set_xlabel("Predicted Rank")
    ax.set_ylabel("Show Title")
    ax.invert_yaxis()  # Highest rank on top
    st.pyplot(fig)